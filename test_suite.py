"""
test_suite.py
-------------
Unit tests for VishingGuard-ZSL prompt validation, router logic, and
LLMEngine behavior (retries / fallback / error handling).

These tests do NOT call the real Groq/OpenAI APIs -- LLMEngine's model
calls are mocked so the suite runs offline, deterministically, and for
free. Run with:

    python -m unittest test_suite.py -v

or, if pytest is available:

    pytest test_suite.py -v
"""

import json
import unittest
from unittest.mock import patch

import prompt_framework as pf
from llm_engine import LLMEngine, route_transcript, _extract_json, compute_fusion_risk, risk_tier
from config import settings


VALID_PAYLOAD = {
    "authority": {"score": 0.9, "evidence": "검찰청 수사관입니다"},
    "urgency": {"score": 0.8, "evidence": "지금 바로"},
    "information_sensitivity": {"score": 0.7, "evidence": "계좌번호를 알려주세요"},
    "threat": {"score": 0.6, "evidence": "체포영장이 발부됩니다"},
    "reward": {"score": 0.0, "evidence": ""},
    "scarcity": {"score": 0.0, "evidence": ""},
    "social_proof": {"score": 0.0, "evidence": ""},
    "final_risk": 0.85,
    "overall_explanation": "High-risk vishing call impersonating a prosecutor.",
    "reasoning_trace": "Caller claims government authority and demands account details urgently.",
}

VISHING_SAMPLE = (
    "여보세요, 서울중앙지검 강력부 수사관입니다. 고객님 명의의 계좌가 대포통장 사기에 연루되어 "
    "긴급히 확인이 필요합니다. 지금 즉시 계좌 비밀번호와 개인정보를 알려주지 않으면 체포영장이 "
    "발부됩니다."
)

BENIGN_SAMPLE = "안녕하세요, 내일 저녁 7시 2명 예약 확인 문의드립니다. 감사합니다."

MIXED_SAMPLE_LONG = (
    "안녕하세요, 배송 관련 문의입니다. " * 20
    + " 참고로 계좌 정지 관련하여 경찰에서 개인정보 확인이 필요하다고 합니다. 긴급히 확인 부탁드립니다."
)


class TestPromptFramework(unittest.TestCase):
    """Tests for prompt_framework.py: schema, versions, message building."""

    def test_required_attributes_present(self):
        self.assertEqual(
            set(pf.REQUIRED_ATTRIBUTES),
            {"authority", "urgency", "information_sensitivity", "threat", "reward", "scarcity", "social_proof"},
        )

    def test_valid_payload_passes_validation(self):
        is_valid, errors = pf.validate_output(VALID_PAYLOAD)
        self.assertTrue(is_valid, msg=f"Unexpected errors: {errors}")
        self.assertEqual(errors, [])

    def test_missing_key_fails_validation(self):
        broken = {k: v for k, v in VALID_PAYLOAD.items() if k != "threat"}
        is_valid, errors = pf.validate_output(broken)
        self.assertFalse(is_valid)
        self.assertTrue(any("threat" in e for e in errors))

    def test_out_of_range_score_fails_validation(self):
        broken = json.loads(json.dumps(VALID_PAYLOAD))  # deep copy
        broken["authority"]["score"] = 1.5
        is_valid, errors = pf.validate_output(broken)
        self.assertFalse(is_valid)
        self.assertTrue(any("authority.score" in e for e in errors))

    def test_non_numeric_final_risk_fails_validation(self):
        broken = json.loads(json.dumps(VALID_PAYLOAD))
        broken["final_risk"] = "high"
        is_valid, errors = pf.validate_output(broken)
        self.assertFalse(is_valid)
        self.assertTrue(any("final_risk" in e for e in errors))

    def test_non_dict_payload_fails_validation(self):
        is_valid, errors = pf.validate_output(["not", "a", "dict"])
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 1)

    def test_all_prompt_versions_load_and_are_nonempty(self):
        for version in pf.list_versions():
            prompt = pf.get_prompt(version)
            self.assertTrue(prompt.system_prompt.strip())
            self.assertTrue(prompt.changelog.strip())

    def test_latest_resolves_to_v3(self):
        self.assertEqual(pf.get_prompt("latest").version, "v3")

    def test_unknown_version_raises(self):
        with self.assertRaises(KeyError):
            pf.get_prompt("v99")

    def test_build_messages_structure(self):
        messages = pf.build_messages(VISHING_SAMPLE, version="v3")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], VISHING_SAMPLE)


class TestRouter(unittest.TestCase):
    """Tests for the complexity router in llm_engine.py."""

    def test_short_benign_routes_to_primary(self):
        self.assertEqual(route_transcript(BENIGN_SAMPLE), settings.primary_model)

    def test_short_vishing_only_routes_to_primary(self):
        # Pure vishing signal, no mixed benign keywords, short -> still cheap/confident case
        short_vishing = "경찰입니다. 계좌 정지 예정이니 개인정보 확인이 긴급합니다."
        self.assertEqual(route_transcript(short_vishing), settings.primary_model)

    def test_mixed_signals_and_long_routes_to_fallback(self):
        self.assertEqual(route_transcript(MIXED_SAMPLE_LONG), settings.fallback_model)

    def test_ambiguity_pattern_contributes_to_routing(self):
        ambiguous = ("확인 필요" * 1) + "계좌 정지 개인정보 긴급 " + ("문의 감사합니다 예약 배송 " * 30)
        # long + mixed keywords + ambiguity marker should push it to fallback
        self.assertEqual(route_transcript(ambiguous), settings.fallback_model)

    def test_router_returns_known_model_name(self):
        result = route_transcript(VISHING_SAMPLE)
        self.assertIn(result, (settings.primary_model, settings.fallback_model))


class TestJsonExtraction(unittest.TestCase):
    """Tests for tolerant JSON parsing of model output."""

    def test_plain_json_parses(self):
        raw = json.dumps(VALID_PAYLOAD)
        parsed = _extract_json(raw)
        self.assertEqual(parsed["final_risk"], 0.85)

    def test_markdown_fenced_json_parses(self):
        raw = "```json\n" + json.dumps(VALID_PAYLOAD) + "\n```"
        parsed = _extract_json(raw)
        self.assertEqual(parsed["final_risk"], 0.85)

    def test_malformed_json_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            _extract_json("{not valid json,,,")


class TestLLMEngine(unittest.TestCase):
    """
    Tests for LLMEngine's control flow: success path, malformed-JSON retry,
    schema-error retry, and full-failure fallback -- all with the network
    calls mocked out.
    """

    def setUp(self):
        self.engine = LLMEngine()

    def test_successful_first_call_returns_payload_with_meta(self):
        with patch.object(self.engine, "_call_model", return_value=json.dumps(VALID_PAYLOAD)) as mock_call:
            result = self.engine.analyze_transcript(VISHING_SAMPLE)
            self.assertEqual(result["final_risk"], 0.85)
            self.assertIn("_meta", result)
            self.assertEqual(result["_meta"]["retries"], 0)
            self.assertIsNone(result["_meta"]["error"])
            mock_call.assert_called_once()

    def test_malformed_json_then_valid_retry_succeeds(self):
        responses = iter(["not json at all", json.dumps(VALID_PAYLOAD)])
        with patch.object(self.engine, "_call_model", side_effect=lambda *a, **k: next(responses)):
            result = self.engine.analyze_transcript(VISHING_SAMPLE)
            self.assertEqual(result["final_risk"], 0.85)
            self.assertEqual(result["_meta"]["retries"], 1)
            self.assertIsNone(result["_meta"]["error"])

    def test_invalid_schema_then_valid_retry_succeeds(self):
        broken = json.loads(json.dumps(VALID_PAYLOAD))
        del broken["threat"]
        responses = iter([json.dumps(broken), json.dumps(VALID_PAYLOAD)])
        with patch.object(self.engine, "_call_model", side_effect=lambda *a, **k: next(responses)):
            result = self.engine.analyze_transcript(VISHING_SAMPLE)
            self.assertEqual(result["final_risk"], 0.85)
            self.assertIsNone(result["_meta"]["error"])

    def test_all_attempts_fail_returns_schema_shaped_error_payload(self):
        with patch.object(self.engine, "_call_model", return_value="still not json"):
            result = self.engine.analyze_transcript(VISHING_SAMPLE)
            # Every required attribute key must still be present so downstream
            # fusion logic never crashes on a missing key.
            for attr in pf.REQUIRED_ATTRIBUTES:
                self.assertIn(attr, result)
            self.assertEqual(result["final_risk"], 0.0)
            self.assertIsNotNone(result["_meta"]["error"])

    def test_empty_transcript_short_circuits_without_calling_model(self):
        with patch.object(self.engine, "_call_model") as mock_call:
            result = self.engine.analyze_transcript("   ")
            mock_call.assert_not_called()
            self.assertEqual(result["_meta"]["error"], "empty_transcript")

    def test_exception_on_primary_falls_back_to_other_model(self):
        primary = settings.primary_model
        fallback = settings.fallback_model

        def side_effect(model_name, messages):
            if model_name == primary:
                raise ConnectionError("simulated network failure")
            return json.dumps(VALID_PAYLOAD)

        with patch.object(self.engine, "_call_model", side_effect=side_effect):
            result = self.engine.analyze_transcript(BENIGN_SAMPLE)  # routes to primary normally
            self.assertEqual(result["_meta"]["model_used"], fallback)
            self.assertTrue(result["_meta"]["fallback_triggered"])
            self.assertIsNone(result["_meta"]["error"])


class TestFusion(unittest.TestCase):
    """Tests for the weighted fusion/decision logic in llm_engine.py."""

    def test_compute_fusion_risk_matches_manual_calculation(self):
        expected = (
            0.9 * settings.fusion_weights["authority"]
            + 0.8 * settings.fusion_weights["urgency"]
            + 0.7 * settings.fusion_weights["information_sensitivity"]
            + 0.6 * settings.fusion_weights["threat"]
            + 0.0 * settings.fusion_weights["reward"]
            + 0.0 * settings.fusion_weights["scarcity"]
            + 0.0 * settings.fusion_weights["social_proof"]
        )
        self.assertAlmostEqual(compute_fusion_risk(VALID_PAYLOAD), round(expected, 4), places=4)

    def test_all_zero_scores_give_zero_risk(self):
        zero_payload = {attr: {"score": 0.0, "evidence": ""} for attr in pf.REQUIRED_ATTRIBUTES}
        self.assertEqual(compute_fusion_risk(zero_payload), 0.0)

    def test_all_max_scores_give_risk_of_one(self):
        max_payload = {attr: {"score": 1.0, "evidence": ""} for attr in pf.REQUIRED_ATTRIBUTES}
        self.assertEqual(compute_fusion_risk(max_payload), 1.0)

    def test_risk_tier_boundaries(self):
        self.assertEqual(risk_tier(0.0), "Low")
        self.assertEqual(risk_tier(0.39), "Low")
        self.assertEqual(risk_tier(0.4), "Moderate")
        self.assertEqual(risk_tier(0.7), "Moderate")
        self.assertEqual(risk_tier(0.71), "High")
        self.assertEqual(risk_tier(1.0), "High")


class TestConfig(unittest.TestCase):
    """Sanity checks on config.py defaults."""

    def test_fusion_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(settings.fusion_weights.values()), 1.0, places=6)

    def test_thresholds_are_ordered(self):
        self.assertLess(settings.moderate_risk_threshold, settings.high_risk_threshold)

    def test_model_names_match_spec(self):
        self.assertEqual(settings.primary_model, "llama3-8b-8192")
        # Fallback now defaults to a free Groq model (no paid API required);
        # OpenAI/gpt-4o-mini remains available as an opt-in via FALLBACK_PROVIDER.
        self.assertEqual(settings.fallback_model, "llama-3.3-70b-versatile")
        self.assertEqual(settings.fallback_provider, "groq")


if __name__ == "__main__":
    unittest.main(verbosity=2)

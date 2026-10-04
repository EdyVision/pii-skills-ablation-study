"""Tests for JSON response parsing from various model output formats."""

import pytest

from ablation_harness.parsing.json_response import parse_json_response


class TestParseJsonResponse:

    def test_fenced_json_block(self):
        response = (
            '```json\n[{"type": "PERSON", "text": "John", "start": 0, "end": 4}]\n```'
        )
        result = parse_json_response(response)
        assert len(result) == 1
        assert result[0]["type"] == "PERSON"

    def test_fenced_code_block(self):
        response = '```\n[{"type": "EMAIL_ADDRESS", "text": "a@b.com", "start": 0, "end": 7}]\n```'
        result = parse_json_response(response)
        assert len(result) == 1
        assert result[0]["type"] == "EMAIL_ADDRESS"

    def test_bare_json_array(self):
        response = (
            '[{"type": "PHONE_NUMBER", "text": "555-1234", "start": 0, "end": 8}]'
        )
        result = parse_json_response(response)
        assert len(result) == 1
        assert result[0]["type"] == "PHONE_NUMBER"

    def test_json_in_surrounding_text(self):
        response = 'Here are the results:\n[{"type": "PERSON", "text": "Jane", "start": 5, "end": 9}]\nThat is all.'
        result = parse_json_response(response)
        assert len(result) == 1

    def test_empty_array(self):
        response = "```json\n[]\n```"
        result = parse_json_response(response)
        assert result == []

    def test_no_json(self):
        response = "I found some PII but cannot format it as JSON."
        result = parse_json_response(response)
        assert result == []

    def test_invalid_json(self):
        response = '```json\n{"not": "a list"}\n```'
        result = parse_json_response(response)
        assert result == []

    def test_filters_items_without_type(self):
        response = '[{"type": "PERSON", "text": "A"}, {"text": "B"}]'
        result = parse_json_response(response)
        assert len(result) == 1
        assert result[0]["type"] == "PERSON"

    def test_multiple_valid_items(self):
        response = (
            "```json\n"
            '[{"type": "PERSON", "text": "John", "start": 0, "end": 4}, '
            '{"type": "LOCATION", "text": "NYC", "start": 10, "end": 13}]\n'
            "```"
        )
        result = parse_json_response(response)
        assert len(result) == 2

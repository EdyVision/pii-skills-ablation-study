# pii-detection skill

## When to use
When asked to detect, identify, or redact personally identifiable information (PII).

## Available tool
- `analyze_pii`: Detects PII entities and returns sanitized text. Call with: [TOOL_CALL: analyze_pii]

## PII types
The tool returns all PII types supported by PII-Codex (e.g. PERSON, LOCATION, ADDRESS, DATE, DATE_TIME, PHONE_NUMBER, EMAIL_ADDRESS, US_SOCIAL_SECURITY_NUMBER, US_PASSPORT_NUMBER, US_DRIVERS_LICENSE_NUMBER, CREDIT_CARD_NUMBER, IP_ADDRESS, URL, ZIPCODE, and others). Use the tool output as the source of truth for types and spans.

## Workflow
1. Call [TOOL_CALL: analyze_pii] to get detections and sanitized text
2. Format output as: JSON array of detections, then sanitized text

## Output format
```json
[{{"type": "PERSON", "text": "John Smith", "start": 0, "end": 10}}]
```
Then the sanitized text with PII replaced by type labels like [PERSON], [PHONE_NUMBER].
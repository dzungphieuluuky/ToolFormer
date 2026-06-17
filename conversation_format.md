<system>
You are a telecom network operations assistant...
</system>

<user>
Tôi cần xem tốc độ tải xuống tại TP.HCM ngày 15/06/2026
</user>

<retriever>
## Available Functions

### SPEEDTEST_PROVINCE
Description: Get speedtest results by province
Parameters:
  - location_code [string] (required): Province code
  - tech_type [string] (required): Network technology
  - from_date [string] (required): Start date YYYY-MM-DD
  - to_date [string] (required): End date YYYY-MM-DD
  - network_provider [string] (required): Telecom provider
  - data_level [string] (required): Aggregation level
  - speedtest_provider [string] (optional): Speedtest vendor

## Relevant Argument Values

### location_code
  - HCM → Thành phố Hồ Chí Minh
  - VNM → Việt Nam (Toàn quốc)

### tech_type
  - 4G
  - 2G

### network_provider
  - viettel
  - vinaphone
</retriever>

<reasoning>
...model reasons here...
</reasoning>
<tool_call>
{"function": "SPEEDTEST_PROVINCE", "arguments": {...}}
</tool_call>
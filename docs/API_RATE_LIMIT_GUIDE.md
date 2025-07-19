# API Rate Limit Guide

## What Happened

You encountered a **429 RESOURCE_EXHAUSTED** error from the Gemini API. This means you've exceeded your current quota for the "GenerateContentPaidTierInputTokensPerModelPerMinute" limit.

## Understanding the Error

```
HTTP/1.1 429 Too Many Requests
You exceeded your current quota, please check your plan and billing details.
```

This happens when:
- You've made too many API calls in a short time period
- Your account has reached its monthly quota limit
- You're on a free tier with lower limits

## Solutions Implemented

### 1. **Retry Logic with Exponential Backoff**
- The system now automatically retries failed requests
- Uses exponential backoff (30s, 60s, 120s delays)
- Handles rate limits gracefully

### 2. **Offline Fallback Mode**
- New command: `generate_offline`
- Generates hypotheses without API calls
- Uses keyword analysis and pattern matching
- Always available when API is rate limited

### 3. **API Status Checker**
- New command: `api_status`
- Tests API connectivity
- Provides specific guidance based on error type

## How to Use the Improved System

### When API is Working
```bash
# Normal hypothesis generation
generate
```

### When API is Rate Limited
```bash
# Check API status
api_status

# Use offline fallback
generate_offline

# Or wait and retry later
generate  # Will automatically retry with delays
```

### Available Commands
- `generate` - Full hypothesis generation with API (with retry logic)
- `generate_offline` - Fallback hypothesis generation without API
- `api_status` - Check API status and get guidance
- `search <query>` - Search and add to package
- `package` - Show current package
- `clear` - Clear package
- `stats` - Show database statistics

## Immediate Solutions

### 1. **Wait for Quota Reset**
- Free tier: Usually resets every minute
- Paid tier: Check your billing dashboard
- Wait 30-60 minutes and try again

### 2. **Use Offline Mode**
```bash
generate_offline
```
This will give you hypotheses based on keyword analysis of your package.

### 3. **Check Your Plan**
- Visit https://ai.google.dev/
- Check your current usage and limits
- Consider upgrading if you need higher limits

### 4. **Reduce Package Size**
- Use `clear` to empty your package
- Add fewer results with `search <query> 100` (limit to 100 results)
- Smaller packages use fewer tokens

## Example Workflow

```bash
# 1. Check API status
api_status

# 2. If rate limited, use offline mode
generate_offline

# 3. Or wait and try again
generate

# 4. If still having issues, work with smaller packages
clear
search "UBR-5" 50  # Limit to 50 results
generate
```

## Prevention Tips

1. **Monitor Package Size**: Keep packages under 1000 chunks
2. **Use Offline Mode**: For quick hypothesis generation
3. **Batch Operations**: Don't make many API calls in quick succession
4. **Check Limits**: Use `api_status` before large operations

## Technical Details

### Rate Limits
- **Free Tier**: ~15 requests per minute
- **Paid Tier**: Higher limits based on plan
- **Token Limits**: ~1M tokens per minute for paid tier

### Retry Strategy
- **Attempts**: 3 retries maximum
- **Backoff**: 30s, 60s, 120s (exponential)
- **Jitter**: Random 0-10s added to prevent thundering herd

### Offline Mode Features
- Keyword extraction from package content
- Pattern-based hypothesis generation
- UBR-5 specific terminology recognition
- No API dependency

## Getting Help

If you continue to have issues:
1. Check your API key in `keys.json`
2. Verify your Google AI Studio account status
3. Consider upgrading your plan
4. Use offline mode for immediate results

The system is now much more robust and will handle rate limits gracefully while providing useful fallback options. 
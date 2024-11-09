```markdown
# Discord Voice AI Assistant

Real-time voice chat bot that joins Discord voice channels and participates in conversations using OpenAI's API.

Forked from:
- [MadcowD's Discord Real-time Assistant Example](https://github.com/MadcowD/ell/blob/main/x/openai_realtime/examples/discord_gpt4o.py)
- [@wgussml](https://x.com/wgussml)


## Features

- Real-time voice interaction using OpenAI's API
- Automatic voice channel joining
- Speech-to-text transcription 
- Natural conversation flow with turn detection
- Automatic conversation prompting during quiet periods
- Custom voice personality/accent support

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Install Opus codec:
- macOS: `brew install opus`
- Ubuntu/Debian: `apt-get install libopus0`
- Windows: `pip install opus-binary`

3. Create `.env` file:
```
OPENAI_API_KEY=your_openai_key
DISCORD_CHANNEL_ID=your_channel_id
DISCORD_BOT_TOKEN=your_bot_token
```

4. Run:
```bash
python main.py
```

## Commands

- `!join` - Join voice channel
- `!leave` - Leave voice channel 
- Mention bot with "join" to make it join your channel

## Requirements

- Python 3.8+
- Discord Bot Token with voice permissions
- OpenAI API key
- Opus codec

## License

MIT
```
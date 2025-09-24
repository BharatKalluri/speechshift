# SpeechShift

A fully local, offline first speech-to-text application made for desktop environments running Wayland compositor (DE's like hyprland etc...). 

Records audio when a hotkey is pressed, transcribes it using faster-whisper, and automatically types the transcribed text.

> This is in beta, since I've been testing only on my system. Please file bug reports if something does not work. Thanks!

## Demo

Demo done on [omarchy](https://omarchy.org/) running hyprland

https://github.com/user-attachments/assets/26a71652-d7e5-4c7f-adf6-217645250771

## Roadmap
- Support for even faster transcription local methods like nvidia parakeet
- Custom vocabulary support
- Use LLM's like ChatGPT to auto format text before pasting

## System Requirements

We'll expand compatibility in the coming days.

- **Window manager**: Wayland
- **Python**: 3.8+
- **Package manager**: UV

## Installation

### 1. Automatic Installation (Recommended)

```bash
uv tool install speechshift
```

Run test to make sure pipewire, wl clipboard is present. It also downloads the whisper (small - ~80mb) model for transcription. 

```bash
speechshift --test
```

Add these lines to your `~/.config/hypr/hyprland.conf`:

The recommended default is Super+Shift+R, but you can set it to anything you like

```bash
# SpeechShift POC Keybinds
bind = SUPER_SHIFT, R, exec, /path/to/speechshift --toggle
```

and setup speechshift daemon to startup on default by adding these lines to `~/.config/hypr/hyprland.conf`

```bash
exec-once = bash -c 'source /path/to/.bashrc && /path/to/speechshift --daemon'
```

we'll need to source the right file, which has the `ASSEMBLYAI_API_KEY` if assembly ai is being used

Then either restart, so that the deamon is automatically run. Or start running the speechshift deamon manually for this session by running

```bash
speechshift --deamon
```

## Usage

1. **Start recording** (Super+Shift+R): You'll see a notification: "🎤 Recording started..."
2. **Stop recording** (Super+Shift+R): Audio is automatically transcribed using faster-whisper or [AssemblyAI](https://www.assemblyai.com/). Transcribed text is typed into the focused window. Notifications show: "🔄 Transcribing audio..." → "✅ Transcribed: [preview]"


## Configuration

SpeechShift can be configured by creating a `config.json` file in `~/.config/speechshift/`. If the file doesn't exist, it will be created with default settings upon first run.

Here's an example configuration to override the default whisper model and language:

```json
{
  "transcription": {
    "engine": "whisper"
  },
  "whisper": {
    "model": "medium",
    "language": "en"
  },
  "audio": {
    "recording_device": null,
    "notification_timeout": 3000
  }
}
```

to use Assembly AI, make sure to set the `ASSEMBLYAI_API_KEY` environment variable and set the transcription engine to `assemblyai`.

_Assembly AI is highly recommended since its much better on accuracy & speed._

## How It Works

### Architecture Overview

```
Keybind (Super+Shift+R)
    ↓
Main Python Script
    ├── PipeWire Audio Recording (sounddevice)
    ├── AI Transcription (faster-whisper)
    ├── Temporary File Management
    ├── Wayland Text Input (wl-clipboard + hyprctl simulate control+V)
    ├── Smart Notifications (notify-send)
```

### Workflow

1. **Keybind Press**: Hyprland detects Super+Shift+R press
2. **Recording Start**:
   - Python script starts PipeWire audio capture
   - Notification: "🎤 Recording started..."
   - Audio streams to temporary WAV file in /tmp
3. **Keybind Release**: Hyprland detects key release
4. **Recording Stop & Transcription**:
   - Audio capture stops
   - Notification: "🔄 Transcribing audio..."
   - faster-whisper transcribes the audio
   - Transcribed text pasted into active window
   - Temporary file automatically deleted
   - Success notification: "✅ Transcribed: [preview]"

### Technical Details

- **Audio Format**: 16-bit WAV, 44.1kHz, mono
- **Transcription Model**: faster-whisper "base" model (configurable)
- **File Handling**: Temporary files in `/tmp`, auto-cleanup after transcription
- **Text Insertion**: Direct typing via wtype, fallback to clipboard paste
- **Notifications**: Smart status updates via notify-send
- **Error Handling**: Graceful fallback with error notifications

## Troubleshooting

### Common Issues

1. **"sounddevice not available"**:

   ```bash
   # Install manually: pip install --user sounddevice numpy
   ```

2. **"Audio recording failed"**:

   - Check PipeWire is running: `systemctl --user status pipewire`
   - Test microphone: `pw-record --list-targets`
   - Verify permissions: ensure user is in `audio` group

3. **"Hyprland socket not found"**:

   - Ensure running under Hyprland
   - Check environment variables: `echo $HYPRLAND_INSTANCE_SIGNATURE`

4. **"Text insertion not working"**:

   - Verify wtype is installed: `wtype --version`
   - Test manually: `wtype "test"`
   - Check focused window accepts text input

5. **"Notifications not showing"**:
   - Test manually: `notify-send "test" "message"`

### Debug Mode

Enable detailed logging by checking `~/.speechshift.log`:

```bash
tail -f ~/.speechshift.log
```

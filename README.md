# Financial Charting

Codex skill for reproducible macroeconomic and financial research charts and
backtest tables built with Python, pandas, and Matplotlib.

## Install

Clone the repository directly into the Codex skills directory:

```powershell
git clone https://github.com/shuang535/financial-charting.git `
  "$env:USERPROFILE\.codex\skills\financial-charting"
```

Restart the Codex session after installation so the skill is discovered.

## Update

```powershell
git -C "$env:USERPROFILE\.codex\skills\financial-charting" pull --ff-only
```

Cloning this public repository does not require a token. Contributors should
use the operating system's Git credential manager for authenticated pushes and
must not store a GitHub token in this repository or in a tracked file.

## Development checks

```powershell
python -X utf8 -m pytest
python -X utf8 <skill-creator-path>\scripts\quick_validate.py .
```

The optional `assets/cathaysite.mplstyle` file provides a chart palette and
typography. Chart helpers do not add a data source or organization
suffix unless the caller supplies one explicitly.

## License

Released under the [MIT License](LICENSE).

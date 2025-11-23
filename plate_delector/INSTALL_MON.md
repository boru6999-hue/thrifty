How to install `mon.traineddata` (Mongolian) for Tesseract

1) Preferred: run the bundled PowerShell installer (as Administrator)

```powershell
# Open PowerShell as Administrator and run:
cd path\to\thrifty\scripts
.\install_mon_traineddata.ps1
```

You can also specify a custom destination tessdata folder:

```powershell
.\install_mon_traineddata.ps1 -Dest "C:\Program Files\Tesseract-OCR\tessdata"
```

2) Manual alternative:
- Download `mon.traineddata` from the Tesseract tessdata repo:
  https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/mon.traineddata
- Copy the downloaded file into your Tesseract tessdata folder, e.g.:
  `C:\Program Files\Tesseract-OCR\tessdata\`

3) Verify installation (PowerShell):

```powershell
# Check tesseract version and available traineddata files
tesseract --version
Get-ChildItem 'C:\Program Files\Tesseract-OCR\tessdata\' | Select-Object Name
```

Notes:
- Copying into `C:\Program Files\...` requires Administrator privileges. Run PowerShell as Administrator.
- After installing `mon.traineddata`, re-run your `car_plate.py` script.
- If you still get no OCR results, share the output of the `Get-ChildItem` command and one sample image so I can help further.

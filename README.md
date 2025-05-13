Es recomana executar-ho en un entorn virtual. 
```bash
python3.11 -m venv .venv-stable
.\.venv-stable\Scripts\activate
$env:GOOGLE_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
pip install -r requirements.txt
```
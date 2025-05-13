Es recomana executar-ho en un entorn virtual. 
```bash
python3.11 -m venv .venv-stable
.\.venv-stable\Scripts\activate
$env:GOOGLE_API_KEY = "YOUR_API_KEY_FROM_GOOGLE_AISTUDIO"
pip install -r requirements.txt
```
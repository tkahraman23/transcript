from flask import Flask, render_template, request
import os
import subprocess
import webbrowser
from threading import Timer
from speech_recognition import Recognizer, AudioFile
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deepmultilingualpunctuation import PunctuationModel

app = Flask(__name__)
yuklenen_dosyalar = "uploads"
app.config["yuklenen_dosyalar"] = yuklenen_dosyalar
dosya_turleri = {"mp3", "m4a"}

tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

punct_model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilingual-sonar-base")

def metin_duzenleme(text):
    text = text.strip()
    try:
        punctuated = punct_model.restore_punctuation(text)
        return punctuated
    except Exception as e:
        print("Noktalama hatası:", e)
        return text

def metin_ozetleme(text):
    girilen_metin = "tr: " + text
    girilenler = tokenizer([girilen_metin], return_tensors="pt", max_length=10000, truncation=True)
    summary_ids = model.generate(
        girilenler["input_ids"],
        max_length=1000,
        min_length=1,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def uygun_turler(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in dosya_turleri

def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.rsplit('.', 1)[0] + '.wav'
    subprocess.run(['ffmpeg', '-i', mp3_path, wav_path, '-y'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def ses_donusturucu(wav_path):
    recognizer = Recognizer()
    with AudioFile(wav_path) as source:
        ses = recognizer.record(source)
    try:
        return recognizer.recognize_google(ses, language="tr-TR")
    except Exception as e:
        return f"Ses tanıma hatası: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "Dosya bulunamadı."
        file = request.files["file"]
        if file.filename == "":
            return "Dosya seçilmedi."
        if file and uygun_turler(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["yuklenen_dosyalar"], filename)
            file.save(file_path)

            wav_path = convert_mp3_to_wav(file_path)
            raw_text = ses_donusturucu(wav_path)

            metin = metin_duzenleme(raw_text)
            ozet = metin_ozetleme(metin)

            return render_template("index.html", metin=metin, ozet=ozet)
    return render_template("index.html")

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    Timer(1, lambda: webbrowser.open(url)).start()
    app.run(debug=True, port=port)
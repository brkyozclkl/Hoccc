from flask import Flask, request, render_template, url_for, send_file, redirect, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
import threading
from PIL import Image
import io
import os
import uuid
import gc
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
from reportlab.graphics.shapes import Drawing, Rect, Line, String
from reportlab.graphics import renderPDF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from reportlab.lib.units import inch
from fpdf import FPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from functools import wraps
from chatbot import PDFChatbot
import json
from models.model_processor import ModelProcessor

# Absolute path to the template directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Güvenli bir anahtar kullanın
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/berko/OneDrive/Masaüstü/hocskin1/hocskin1/hocskin/HocSkin/instance/hocskin.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Kullanıcı modeli
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    skin_type = db.Column(db.String(20))
    skin_concerns = db.Column(db.String(200))
    allergies = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    analyses = db.relationship('Analysis', backref='user', lazy=True)

# Analiz modeli
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    image_path = db.Column(db.String(200), nullable=False)
    acne_count = db.Column(db.Integer, default=0)
    wrinkle_count = db.Column(db.Integer, default=0)
    eyebag_count = db.Column(db.Integer, default=0)
    redness_count = db.Column(db.Integer, default=0)
    skin_type = db.Column(db.String(20))  # dry, normal, oily
    skin_age = db.Column(db.Float)
    skin_age_assessment = db.Column(db.String(500))
    recommendations = db.relationship('ProductRecommendation', back_populates='analysis', lazy=True)

class ProductRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis.id'), nullable=True)  # Analiz ID'si opsiyonel
    product_name = db.Column(db.String(200), nullable=False)
    brand = db.Column(db.String(100))
    product_type = db.Column(db.String(50))  # cleanser, moisturizer, serum, etc.
    description = db.Column(db.Text)
    price_range = db.Column(db.String(50))  # low, medium, high
    ingredients = db.Column(db.Text)
    usage_instructions = db.Column(db.Text)
    target_skin_type = db.Column(db.String(20))
    is_active = db.Column(db.Boolean, default=True)
    analysis = db.relationship('Analysis', back_populates='recommendations')

# Veritabanını oluştur
with app.app_context():
    db.create_all()

# Model işlemcisini oluştur
model_processor = ModelProcessor()

# Çıktı dizini
OUTPUT_DIR = "static/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maksimum görüntü boyutu
MAX_IMAGE_SIZE = (400, 400)

# Register Arial font which has good Turkish character support
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))

# Global chatbot instance
chatbot = None

def resize_image(image):
    """Görüntüyü maksimum boyuta sığacak şekilde yeniden boyutlandır"""
    if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
        image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    return image

def process_single_model(model, image, key, color, draw):
    """Tek bir modeli çalıştır ve sonuçları döndür"""
    try:
        # Görüntüyü numpy dizisine çevir ve bellek optimizasyonu yap
        img_array = np.array(image, dtype=np.uint8)
        
        # Model çıktılarını kısıtla
        results = model(img_array, verbose=False, conf=0.25)
        detections = results[0].boxes

        if key == 'skin_age':
            age_value = results[0].names[int(detections.cls[0])] if len(detections) > 0 else "Tespit edilemedi"
            result = {
                "count": len(detections),
                "age": age_value
            }
        else:
            for box in detections.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1 - 10), key.capitalize(), fill=color)

            result = {
                "count": len(detections)
            }
        
        # Belleği temizle
        del results
        del detections
        del img_array
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"Hata oluştu ({key}): {str(e)}")
        return {"count": 0, "error": str(e)}

def calculate_skin_age(results):
    """Cilt analiz sonuçlarına göre cilt yaşını hesaplar"""
    base_age = results.get('age', {}).get('value', 30)  # Gerçek yaş veya varsayılan 30
    skin_age = base_age
    
    # Akne (Sivilce) değerlendirmesi
    if 'acne' in results:
        acne_count = results['acne'].get('count', 0)
        if acne_count > 10:
            skin_age -= 2  # Genç görünüm
        elif acne_count > 5:
            skin_age -= 1
        elif acne_count == 0:
            skin_age += 1  # Olgun cilt
    
    # Kırışıklık değerlendirmesi
    if 'wrinkle' in results:
        wrinkle_count = results['wrinkle'].get('count', 0)
        if wrinkle_count > 15:
            skin_age += 5  # İleri yaş belirtisi
        elif wrinkle_count > 10:
            skin_age += 3
        elif wrinkle_count > 5:
            skin_age += 2
        elif wrinkle_count > 0:
            skin_age += 1
    
    # Göz altı torbaları değerlendirmesi
    if 'eyebag' in results:
        eyebag_count = results['eyebag'].get('count', 0)
        if eyebag_count > 3:
            skin_age += 3
        elif eyebag_count > 0:
            skin_age += 1
    
    # Kızarıklık değerlendirmesi
    if 'redness' in results:
        redness_count = results['redness'].get('count', 0)
        if redness_count > 5:
            skin_age += 2  # Hassas/yaşlı cilt belirtisi
        elif redness_count > 0:
            skin_age += 1
    
    # Cilt yaşını 18-80 arasında sınırla
    skin_age = max(18, min(80, skin_age))
    
    # Yaş farkını hesapla
    age_difference = skin_age - base_age
    
    # Değerlendirme metni oluştur
    if age_difference < -3:
        assessment = "Cildiniz gerçek yaşınızdan daha genç görünüyor!"
    elif age_difference < 0:
        assessment = "Cildiniz gerçek yaşınıza yakın ve genç görünüyor."
    elif age_difference == 0:
        assessment = "Cildiniz gerçek yaşınızla uyumlu."
    elif age_difference < 3:
        assessment = "Cildiniz gerçek yaşınızdan biraz daha olgun görünüyor."
    else:
        assessment = "Cildiniz gerçek yaşınızdan daha olgun görünüyor."
    
    return {
        'value': round(skin_age, 1),
        'base_age': base_age,
        'difference': age_difference,
        'assessment': assessment
    }

def detect_age(image):
    """Görüntüden yaş tespiti yap"""
    try:
        # Görüntüyü numpy dizisine çevir
        img_array = np.array(image)
        
        # OpenCV formatına çevir (BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Boyutlandır ve normalize et
        img_array = cv2.resize(img_array, (200, 200))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = img_array / 255.0
        
        # Batch dimension ekle
        img_array = np.expand_dims(img_array, axis=0)
        
        # Tahmin yap
        predicted_age = age_model.predict(img_array)[0][0]
        
        return round(predicted_age, 1)
    except Exception as e:
        print(f"Yaş tespiti hatası: {str(e)}")
        return None

# Ana sayfa (form ekranı)
@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['GET'])
def analyze_page():
    return render_template('analyze.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Ücretsiz deneme kontrolü
    has_free_trial = session.get('has_free_trial', True)
    is_logged_in = 'user_id' in session

    if not is_logged_in and not has_free_trial:
        flash('Ücretsiz deneme hakkınızı kullandınız. Daha fazla analiz için lütfen üye olun.')
        return redirect(url_for('register'))

    image = None
    
    try:
        # Kamera ile çekilen fotoğrafı kontrol et
        if 'capturedImage' in request.form and request.form['capturedImage']:
            try:
                import base64
                image_data = request.form['capturedImage'].split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                return f"Kamera görüntüsü işlenemedi: {e}", 400
        
        # Dosya yüklemesini kontrol et
        elif 'image' in request.files and request.files['image'].filename:
            try:
                file = request.files['image']
                image = Image.open(file.stream).convert("RGB")
            except Exception as e:
                return f"Görsel işlenemedi: {e}", 400
        
        else:
            return "Görsel yüklenmedi", 400

        # Görüntüyü yeniden boyutlandır
        image = resize_image(image)
        
        # Model işlemcisini kullanarak analiz yap
        results, img_copy = model_processor.analyze_skin(image)
        
        # Sonuç görselini kaydet
        filename = f"combined_{uuid.uuid4().hex}.jpg"
        relative_path = f"results/{filename}"
        absolute_path = os.path.join(app.static_folder, "results", filename)
        img_copy.save(absolute_path, quality=85, optimize=True)
        result_image_path = relative_path

        # Son bellek temizliği
        del image
        del img_copy
        gc.collect()

        # Cilt yaşı analizi yap
        skin_age = calculate_skin_age(results)
        results['skin_age'] = skin_age

        # Eğer kullanıcı giriş yapmışsa veritabanına kaydet
        if is_logged_in:
            analysis = Analysis(
                user_id=session['user_id'],
                image_path=result_image_path,
                acne_count=results.get('acne', {}).get('count', 0),
                wrinkle_count=results.get('wrinkle', {}).get('count', 0),
                eyebag_count=results.get('eyebag', {}).get('count', 0),
                redness_count=results.get('redness', {}).get('count', 0),
                skin_type=results.get('skin_type', 'unknown'),
                skin_age=skin_age['value'],
                skin_age_assessment=skin_age['assessment']
            )
            db.session.add(analysis)
            db.session.commit()
            
            # Ürün önerilerini oluştur ve kaydet
            recommendations = generate_product_recommendations(analysis)
            for rec in recommendations:
                db.session.add(rec)
            db.session.commit()
            
            # Önerileri results dictionary'sine ekle
            results['recommendations'] = recommendations
        else:
            # Ücretsiz deneme hakkını kullan
            session['has_free_trial'] = False

        return render_template("results.html", 
                            results=results, 
                            result_image_path=result_image_path,
                            is_logged_in=is_logged_in,
                            is_free_trial=not is_logged_in)
    
    except Exception as e:
        print(f"Genel hata: {str(e)}")
        return f"İşlem sırasında bir hata oluştu: {str(e)}", 500

@app.route('/about')
def about():
    return render_template("about.html")

def generate_pdf_report(results, image_path, user=None):
    """Detaylı PDF raporu oluşturur"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hocskin_rapor_{timestamp}.pdf"
    # Mutlak yol ile kaydet
    abs_report_dir = os.path.join(app.static_folder, 'reports')
    os.makedirs(abs_report_dir, exist_ok=True)
    abs_filepath = os.path.join(abs_report_dir, filename)
    
    # PDF oluştur
    doc = SimpleDocTemplate(
        abs_filepath,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    # Özel stiller oluştur
    styles = getSampleStyleSheet()
    
    # Modern başlık stili
    styles.add(ParagraphStyle(
        name='HocSkinTitle',
        parent=styles['Title'],
        fontSize=32,
        spaceAfter=20,
        spaceBefore=20,
        alignment=1,
        fontName='Arial-Bold',
        textColor=colors.HexColor('#2C3E50'),
        leading=40
    ))
    
    # Alt başlık stili
    styles.add(ParagraphStyle(
        name='HocSkinHeading',
        parent=styles['Heading2'],
        fontSize=20,
        spaceBefore=25,
        spaceAfter=15,
        fontName='Arial-Bold',
        textColor=colors.HexColor('#34495E'),
        borderPadding=10,
        borderWidth=1,
        borderColor=colors.HexColor('#BDC3C7'),
        borderRadius=5,
        leading=24,
        keepWithNext=True
    ))
    
    # Alt başlık stili (sayfa kesmesiz)
    styles.add(ParagraphStyle(
        name='HocSkinHeadingNoPB',
        parent=styles['HocSkinHeading'],
        pageBreakBefore=False
    ))
    
    # Normal metin stili
    styles.add(ParagraphStyle(
        name='HocSkinText',
        parent=styles['Normal'],
        fontSize=12,
        leading=18,
        fontName='Arial',
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=12,
        alignment=0
    ))
    
    # Bilgi kutusu stili
    styles.add(ParagraphStyle(
        name='HocSkinInfoBox',
        parent=styles['Normal'],
        fontSize=12,
        leading=18,
        fontName='Arial',
        textColor=colors.HexColor('#2C3E50'),
        backColor=colors.HexColor('#ECF0F1'),
        borderPadding=15,
        borderWidth=1,
        borderColor=colors.HexColor('#BDC3C7'),
        borderRadius=5,
        spaceAfter=20
    ))
    
    story = []
    
    # Logo ve başlık bölümü
    header_elements = []
    logo_path = os.path.join('static', 'img', 'logo.png')
    if os.path.exists(logo_path):
        logo = ReportLabImage(logo_path, width=2*inch, height=2*inch)
        header_elements.append(logo)
    
    header_elements.append(Spacer(1, 20))
    header_elements.append(Paragraph("HocSkin Cilt Analiz Raporu", styles['HocSkinTitle']))
    
    # Başlık bölümünü bir arada tut
    for element in header_elements:
        story.append(element)
    
    # Kullanıcı bilgileri bölümü
    if user:
        user_info = f"""
        <para backColor="#F8F9F9" borderPadding="20" borderWidth="1" borderColor="#BDC3C7" borderRadius="5">
        <b><font size="14" color="#34495E">Kullanıcı Bilgileri</font></b>
        <br/><br/>
        <b>Kullanıcı Adı:</b> {user.username}<br/>
        <b>E-posta:</b> {user.email}<br/>
        <b>Analiz Tarihi:</b> {datetime.now().strftime("%d/%m/%Y %H:%M")}
        </para>
        """
        story.append(Paragraph(user_info, styles['HocSkinInfoBox']))
    
    # Analiz görseli bölümü
    story.append(Paragraph("Analiz Görseli", styles['HocSkinHeadingNoPB']))
    # image_path 'results/filename' ise tam yolu oluştur
    img_full_path = os.path.join(app.static_folder, image_path.replace('/', os.sep))
    if not os.path.exists(img_full_path):
        # Eğer görsel yoksa placeholder ekle veya hata mesajı
        story.append(Paragraph("<b>Analiz görseli bulunamadı.</b>", styles['HocSkinText']))
    else:
        img = ReportLabImage(img_full_path, width=6.5*inch, height=4.5*inch)
        story.append(img)
    story.append(Spacer(1, 20))
    
    # Cilt analizi sonuçları bölümü
    story.append(Paragraph("Cilt Analizi Sonuçları", styles['HocSkinHeading']))
    analysis_info = f"""
    <para backColor="#F8F9F9" borderPadding="20" borderWidth="1" borderColor="#BDC3C7" borderRadius="5">
    <b><font color="#34495E">Cilt Tipi:</font></b> {results.get('skin_type', 'Belirlenemedi')}<br/><br/>
    <b><font color="#34495E">Tahmini Yaş:</font></b> {results.get('age', {}).get('value', 'Belirlenemedi')} yaş<br/><br/>
    <b><font color="#34495E">Cilt Yaşı:</font></b> {results.get('skin_age', {}).get('value', 'Belirlenemedi')} yaş<br/><br/>
    <b><font color="#34495E">Değerlendirme:</font></b><br/>
    {results.get('skin_age', {}).get('assessment', 'Değerlendirme yapılamadı')}
    </para>
    """
    story.append(Paragraph(analysis_info, styles['HocSkinInfoBox']))
    
    # Cilt problemleri bölümü
    story.append(Paragraph("Tespit Edilen Cilt Sorunları", styles['HocSkinHeading']))
    
    def severity_text(count):
        if count < 3: return "Düşük"
        elif count < 6: return "Orta"
        else: return "Yüksek"
    
    def severity_color(severity):
        colors = {
            "Düşük": "#2ECC71",
            "Orta": "#F1C40F",
            "Yüksek": "#E74C3C"
        }
        return colors.get(severity, "#2C3E50")
    
    problems_container = []
    if any(results.get(key, {}).get('count', 0) > 0 for key in ['acne', 'wrinkle', 'eyebag', 'redness']):
        if results.get('acne', {}).get('count', 0) > 0:
            count = results['acne']['count']
            sev = severity_text(count)
            problems_container.append(
                Paragraph(
                    f"""<para backColor="#F8F9F9" borderPadding="15" borderWidth="1" 
                    borderColor="#BDC3C7" borderRadius="5">
                    • <b>Sivilce:</b> {count} adet tespit edildi 
                    <font color="{severity_color(sev)}"><b>(Şiddet: {sev})</b></font>
                    </para>""",
                    styles['HocSkinText']
                )
            )
        
        if results.get('wrinkle', {}).get('count', 0) > 0:
            count = results['wrinkle']['count']
            sev = severity_text(count)
            problems_container.append(
                Paragraph(
                    f"""<para backColor="#F8F9F9" borderPadding="15" borderWidth="1" 
                    borderColor="#BDC3C7" borderRadius="5">
                    • <b>Kırışıklık:</b> {count} adet tespit edildi 
                    <font color="{severity_color(sev)}"><b>(Şiddet: {sev})</b></font>
                    </para>""",
                    styles['HocSkinText']
                )
            )
        
        if results.get('eyebag', {}).get('count', 0) > 0:
            count = results['eyebag']['count']
            sev = severity_text(count)
            problems_container.append(
                Paragraph(
                    f"""<para backColor="#F8F9F9" borderPadding="15" borderWidth="1" 
                    borderColor="#BDC3C7" borderRadius="5">
                    • <b>Göz Altı Torbası:</b> {count} adet tespit edildi 
                    <font color="{severity_color(sev)}"><b>(Şiddet: {sev})</b></font>
                    </para>""",
                    styles['HocSkinText']
                )
            )
        
        if results.get('redness', {}).get('count', 0) > 0:
            count = results['redness']['count']
            sev = severity_text(count)
            problems_container.append(
                Paragraph(
                    f"""<para backColor="#F8F9F9" borderPadding="15" borderWidth="1" 
                    borderColor="#BDC3C7" borderRadius="5">
                    • <b>Kızarıklık:</b> {count} bölge tespit edildi 
                    <font color="{severity_color(sev)}"><b>(Şiddet: {sev})</b></font>
                    </para>""",
                    styles['HocSkinText']
                )
            )
    else:
        problems_container.append(
            Paragraph(
                """<para backColor="#F8F9F9" borderPadding="15" borderWidth="1" 
                borderColor="#BDC3C7" borderRadius="5">
                Herhangi bir cilt sorunu tespit edilmedi.
                </para>""",
                styles['HocSkinText']
            )
        )
    
    # Cilt problemlerini bir arada tut
    for element in problems_container:
        story.append(element)
        story.append(Spacer(1, 10))
    
    # Ürün önerileri bölümü
    if results.get('recommendations'):
        story.append(Paragraph("Kişiselleştirilmiş Ürün Önerileri", styles['HocSkinHeading']))
        for rec in results['recommendations']:
            rec_text = f"""
            <para backColor="#F8F9F9" borderPadding="20" borderWidth="1" 
            borderColor="#BDC3C7" borderRadius="5">
            <b><font size="14" color="#2980B9">{rec.get('product_name', 'Ürün adı belirtilmemiş')}</font></b><br/>
            <i><font color="#7F8C8D">{rec.get('brand', 'Marka belirtilmemiş')}</font></i><br/><br/>
            <font color="#34495E">{rec.get('description', 'Açıklama belirtilmemiş')}</font><br/><br/>
            <b>Fiyat Aralığı:</b> {rec.get('price_range', 'Belirtilmemiş')}<br/>
            <b>İçerikler:</b> {rec.get('ingredients', 'Belirtilmemiş')}<br/>
            <b>Kullanım:</b> {rec.get('usage_instructions', 'Belirtilmemiş')}
            </para>
            """
            story.append(Paragraph(rec_text, styles['HocSkinInfoBox']))
            story.append(Spacer(1, 15))
    
    # Altbilgi
    story.append(Spacer(1, 30))
    footer_text = f"""
    <para alignment="center" backColor="#2C3E50" textColor="white" 
    borderPadding="15" borderRadius="5">
    <font color="white">
    Bu rapor HocSkin tarafından {datetime.now().strftime("%d/%m/%Y")} tarihinde oluşturulmuştur.<br/>
    www.hocskin.com
    </font>
    </para>
    """
    story.append(Paragraph(footer_text, styles['HocSkinText']))
    
    # Raporu oluştur
    doc.build(story)
    return filename  # Sadece dosya adını döndür

@app.route('/download_report')
def download_report():
    """PDF raporunu indir"""
    if 'user_id' not in session:
        flash('Rapor indirmek için giriş yapmalısınız.')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        flash('Kullanıcı bulunamadı.')
        return redirect(url_for('login'))
    
    # Son analizi al
    analysis = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.date.desc()).first()
    if not analysis:
        flash('Analiz bulunamadı.')
        return redirect(url_for('history'))
    
    # Sonuçları hazırla
    results = {
        'age': {'value': analysis.skin_age},
        'skin_age': {'value': analysis.skin_age, 'assessment': analysis.skin_age_assessment},
        'acne': {'count': analysis.acne_count},
        'wrinkle': {'count': analysis.wrinkle_count},
        'eyebag': {'count': analysis.eyebag_count},
        'redness': {'count': analysis.redness_count},
        'recommendations': [rec.__dict__ for rec in analysis.recommendations]
    }
    
    # PDF raporu oluştur
    filename = generate_pdf_report(results, analysis.image_path, user)
    abs_filepath = os.path.join(app.static_folder, 'reports', filename)
    if not os.path.exists(abs_filepath):
        flash('Rapor dosyası oluşturulamadı!')
        return redirect(url_for('history'))
    return send_file(
        abs_filepath,
        as_attachment=True,
        download_name=filename
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Şifreler eşleşmiyor!')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Bu kullanıcı adı zaten kullanılıyor!')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Bu e-posta adresi zaten kullanılıyor!')
            return redirect(url_for('register'))

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        flash('Kayıt başarılı! Şimdi giriş yapabilirsiniz.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            user.last_login = datetime.utcnow()
            db.session.commit()
            flash('Giriş başarılı!')
            return redirect(url_for('index'))
        else:
            flash('E-posta veya şifre hatalı!')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Çıkış yapıldı!')
    return redirect(url_for('index'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Lütfen giriş yapın!')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.date.desc()).all()
    return render_template('history.html', analyses=analyses)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Lütfen giriş yapın!')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.date.desc()).all()
    
    # Analiz istatistikleri
    total_analyses = len(analyses)
    last_analysis = analyses[0] if analyses else None
    
    # Ortalama değerler
    avg_acne = sum(a.acne_count for a in analyses) / total_analyses if total_analyses > 0 else 0
    avg_wrinkle = sum(a.wrinkle_count for a in analyses) / total_analyses if total_analyses > 0 else 0
    avg_eyebag = sum(a.eyebag_count for a in analyses) / total_analyses if total_analyses > 0 else 0
    avg_redness = sum(a.redness_count for a in analyses) / total_analyses if total_analyses > 0 else 0
    
    return render_template('profile.html', 
                         user=user,
                         analyses=analyses,
                         total_analyses=total_analyses,
                         last_analysis=last_analysis,
                         avg_acne=avg_acne,
                         avg_wrinkle=avg_wrinkle,
                         avg_eyebag=avg_eyebag,
                         avg_redness=avg_redness)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        flash('Lütfen giriş yapın!')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:
        flash('Kullanıcı bulunamadı!')
        return redirect(url_for('login'))

    try:
        user.age = request.form.get('age', type=int)
        user.gender = request.form.get('gender')
        user.skin_type = request.form.get('skin_type')
        user.skin_concerns = request.form.get('skin_concerns')
        user.allergies = request.form.get('allergies')

        db.session.commit()
        flash('Profil başarıyla güncellendi!')
    except Exception as e:
        db.session.rollback()
        flash('Profil güncellenirken bir hata oluştu!')
        print(f"Profil güncelleme hatası: {str(e)}")

    return redirect(url_for('profile'))

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        flash('Lütfen giriş yapın!')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:
        flash('Kullanıcı bulunamadı!')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            user.username = request.form.get('username', user.username)
            user.email = request.form.get('email', user.email)
            user.age = request.form.get('age', type=int)
            user.gender = request.form.get('gender')
            user.skin_type = request.form.get('skin_type')
            user.skin_concerns = request.form.get('skin_concerns')
            user.allergies = request.form.get('allergies')

            # Şifre değişikliği kontrolü
            new_password = request.form.get('new_password')
            if new_password:
                if len(new_password) < 6:
                    flash('Şifre en az 6 karakter olmalıdır!')
                    return redirect(url_for('edit_profile'))
                user.password_hash = generate_password_hash(new_password)

            db.session.commit()
            flash('Profil başarıyla güncellendi!')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            flash('Profil güncellenirken bir hata oluştu!')
            print(f"Profil güncelleme hatası: {str(e)}")

    return render_template('edit_profile.html', user=user)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        sender_email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        try:
            # E-posta gönderme işlemi
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = 'hoctechkurumsal@gmail.com'
            msg['Subject'] = f'İletişim Formu: {subject}'
            
            body = f"""
            İsim: {name}
            E-posta: {sender_email}
            Konu: {subject}
            
            Mesaj:
            {message}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # SMTP sunucusu üzerinden e-posta gönderme
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login('hoctechkurumsal@gmail.com', 'qtdt qjkj ihfj thmi')
            server.send_message(msg)
            server.quit()
            
            flash('Mesajınız başarıyla gönderilmiştir. Ekiplerimiz en yakın zamanda sizinle iletişime geçecektir.', 'success')
            return redirect(url_for('contact'))
        except Exception as e:
            flash('Mesaj gönderilirken bir hata oluştu! Lütfen daha sonra tekrar deneyiniz.', 'error')
            print(f"E-posta gönderme hatası: {str(e)}")
            return redirect(url_for('contact'))
            
    return render_template('contact.html')

def generate_product_recommendations(analysis):
    """Cilt analizi sonuçlarına göre ürün önerileri oluştur"""
    recommendations = []
    
    # Sivilce (acne) için öneriler
    if analysis.acne_count > 0:
        # Doğal öneriler
        recommendations.extend([
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Doğal Sivilce Bakımı",
                brand="Doğal Çözüm",
                product_type="natural_remedy",
                description="Sivilce sorunları için doğal çözümler",
                price_range="low",
                ingredients="Çay ağacı yağı, Aloe vera, Bal, Yeşil çay, Elma sirkesi, Zerdeçal, Limon suyu",
                usage_instructions="""Kullanım Bölgeleri: Yüz, sırt, göğüs
1. Çay ağacı yağı: 1 damla çay ağacı yağını 9 damla su ile seyreltin. Pamukla sivilceli bölgelere uygulayın.
2. Aloe vera: Taze aloe vera jeli sivilceli bölgelere sürün, 20 dakika bekletin.
3. Bal maskesi: 1 yemek kaşığı balı sivilceli bölgelere sürün, 15 dakika bekletin.
4. Yeşil çay toniği: Demlenmiş yeşil çayı soğutun, pamukla sivilceli bölgelere uygulayın.
5. Elma sirkesi toniği: 1 ölçü elma sirkesi, 3 ölçü su ile karıştırın. Pamukla uygulayın.
6. Zerdeçal maskesi: 1 çay kaşığı zerdeçal, 1 yemek kaşığı bal ile karıştırın. 15 dakika bekletin.
7. Limon suyu: Taze limon suyunu pamukla sivilceli bölgelere uygulayın."""
            )
        ])
        
        # Ürün önerileri
        recommendations.extend([
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Salicylic Acid Cleanser",
                brand="CeraVe",
                product_type="cleanser",
                description="Salisilik asit içeren temizleyici, gözenekleri temizler ve sivilce oluşumunu engeller.",
                price_range="medium",
                ingredients="Salicylic Acid, Niacinamide, Ceramides",
                usage_instructions="Kullanım Bölgeleri: Yüz, sırt, göğüs\nSabah ve akşam temizleyici olarak kullanın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Niacinamide Serum",
                brand="The Ordinary",
                product_type="serum",
                description="Niasinamid içeren serum, cilt bariyerini güçlendirir ve iltihabı azaltır.",
                price_range="low",
                ingredients="Niacinamide, Zinc",
                usage_instructions="Kullanım Bölgeleri: Yüz\nTemiz cilde günde 1-2 kez uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Benzoyl Peroxide Treatment",
                brand="La Roche-Posay",
                product_type="treatment",
                description="Benzoyl peroxide içeren tedavi ürünü, sivilce bakterilerini öldürür.",
                price_range="medium",
                ingredients="Benzoyl Peroxide, Glycerin, Niacinamide",
                usage_instructions="Kullanım Bölgeleri: Sivilceli bölgeler\nAkşamları temiz cilde uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Tea Tree Oil Spot Treatment",
                brand="The Body Shop",
                product_type="spot_treatment",
                description="Çay ağacı yağı içeren nokta tedavisi, sivilceleri kurutur ve iltihabı azaltır.",
                price_range="low",
                ingredients="Tea Tree Oil, Witch Hazel, Aloe Vera",
                usage_instructions="Kullanım Bölgeleri: Sivilceli bölgeler\nGünde 2-3 kez uygulayın."
            )
        ])
    
    # Kırışıklık (wrinkle) için öneriler
    if analysis.wrinkle_count > 0:
        # Doğal öneriler
        recommendations.extend([
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Doğal Kırışıklık Bakımı",
                brand="Doğal Çözüm",
                product_type="natural_remedy",
                description="Kırışıklıklar için doğal çözümler",
                price_range="low",
                ingredients="Avokado, Hindistan cevizi yağı, Yumurta akı, Bal, Zeytinyağı, Yumurta sarısı, Muz, Papaya",
                usage_instructions="""Kullanım Bölgeleri: Yüz, boyun, dekolte
1. Avokado maskesi: Olgun avokadoyu ezin, 15 dakika yüzünüze uygulayın.
2. Hindistan cevizi yağı: Gece yatmadan önce kırışıklıklı bölgelere masaj yaparak uygulayın.
3. Yumurta akı maskesi: Yumurta akını çırpın, 15 dakika yüzünüze uygulayın.
4. Bal maskesi: Balı kırışıklıklı bölgelere sürün, 20 dakika bekletin.
5. Zeytinyağı masajı: Birkaç damla zeytinyağını kırışıklıklı bölgelere masaj yaparak uygulayın.
6. Yumurta sarısı maskesi: Yumurta sarısını çırpın, 15 dakika yüzünüze uygulayın.
7. Muz maskesi: Olgun muzu ezin, 15 dakika yüzünüze uygulayın.
8. Papaya maskesi: Papaya püresini yüzünüze uygulayın, 15 dakika bekletin."""
            )
        ])
        
        # Ürün önerileri
        recommendations.extend([
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Retinol Cream",
                brand="La Roche-Posay",
                product_type="moisturizer",
                description="Retinol içeren krem, kırışıklıkları azaltır ve cilt yenilenmesini destekler.",
                price_range="high",
                ingredients="Retinol, Hyaluronic Acid, Ceramides",
                usage_instructions="Kullanım Bölgeleri: Yüz, boyun, dekolte\nAkşamları temiz cilde uygulayın. Güneş koruyucu kullanmayı unutmayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Vitamin C Serum",
                brand="SkinCeuticals",
                product_type="serum",
                description="C vitamini serumu, kollajen üretimini artırır ve cilt tonunu eşitler.",
                price_range="high",
                ingredients="Vitamin C, Ferulic Acid, Vitamin E",
                usage_instructions="Kullanım Bölgeleri: Yüz, boyun\nSabahları temiz cilde uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Peptide Complex",
                brand="The Ordinary",
                product_type="serum",
                description="Peptit kompleksi, kırışıklıkları azaltır ve cilt elastikiyetini artırır.",
                price_range="medium",
                ingredients="Matrixyl, Argireline, Hyaluronic Acid",
                usage_instructions="Kullanım Bölgeleri: Yüz, boyun\nGünde 2 kez temiz cilde uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Hyaluronic Acid Serum",
                brand="Vichy",
                product_type="serum",
                description="Hiyalüronik asit serumu, cildi derinlemesine nemlendirir ve kırışıklıkları azaltır.",
                price_range="medium",
                ingredients="Hyaluronic Acid, Vitamin C, Vitamin E",
                usage_instructions="Kullanım Bölgeleri: Yüz, boyun\nGünde 2 kez temiz cilde uygulayın."
            )
        ])
    
    # Göz altı torbaları (eyebag) için öneriler
    if analysis.eyebag_count > 0:
        # Doğal öneriler
        recommendations.extend([
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Doğal Göz Altı Bakımı",
                brand="Doğal Çözüm",
                product_type="natural_remedy",
                description="Göz altı torbaları için doğal çözümler",
                price_range="low",
                ingredients="Salatalık, Patates, Yeşil çay, Gül suyu, Buz küpleri, Elma sirkesi, Aloe vera, Hindistan cevizi yağı",
                usage_instructions="""Kullanım Bölgeleri: Göz çevresi
1. Salatalık dilimleri: Soğutulmuş salatalık dilimlerini gözlerinizin üzerine koyun, 10-15 dakika bekletin.
2. Patates maskesi: Rendelenmiş patatesi göz altlarına uygulayın, 15 dakika bekletin.
3. Yeşil çay poşetleri: Kullanılmış ve soğutulmuş yeşil çay poşetlerini gözlerinizin üzerine koyun.
4. Gül suyu: Pamukla göz çevresine gül suyu uygulayın.
5. Buz masajı: Temiz bir beze sarılı buz küplerini göz çevresine masaj yaparak uygulayın.
6. Elma sirkesi toniği: 1 ölçü elma sirkesi, 3 ölçü su ile karıştırın. Pamukla göz çevresine uygulayın.
7. Aloe vera jeli: Taze aloe vera jeli göz altlarına uygulayın, 15 dakika bekletin.
8. Hindistan cevizi yağı masajı: Birkaç damla hindistan cevizi yağını göz çevresine masaj yaparak uygulayın."""
            )
        ])
        
        # Ürün önerileri
        recommendations.extend([
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Caffeine Eye Cream",
                brand="The Inkey List",
                product_type="eye_cream",
                description="Kafein içeren göz kremi, göz altı torbalarını azaltır ve şişkinliği giderir.",
                price_range="low",
                ingredients="Caffeine, Hyaluronic Acid, Peptides",
                usage_instructions="Kullanım Bölgeleri: Göz çevresi\nSabah ve akşam göz çevresine uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Retinol Eye Cream",
                brand="La Roche-Posay",
                product_type="eye_cream",
                description="Retinol içeren göz kremi, göz çevresi kırışıklıklarını azaltır.",
                price_range="medium",
                ingredients="Retinol, Caffeine, Hyaluronic Acid",
                usage_instructions="Kullanım Bölgeleri: Göz çevresi\nAkşamları temiz cilde uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Vitamin C Eye Cream",
                brand="Kiehl's",
                product_type="eye_cream",
                description="C vitamini içeren göz kremi, göz çevresi koyu halkalarını azaltır.",
                price_range="high",
                ingredients="Vitamin C, Hyaluronic Acid, Caffeine",
                usage_instructions="Kullanım Bölgeleri: Göz çevresi\nSabah ve akşam göz çevresine uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Hyaluronic Acid Eye Gel",
                brand="The Ordinary",
                product_type="eye_gel",
                description="Hiyalüronik asit içeren göz jeli, göz çevresini nemlendirir ve şişkinliği azaltır.",
                price_range="low",
                ingredients="Hyaluronic Acid, Caffeine, Peptides",
                usage_instructions="Kullanım Bölgeleri: Göz çevresi\nGünde 2-3 kez göz çevresine uygulayın."
            )
        ])
    
    # Kızarıklık (redness) için öneriler
    if analysis.redness_count > 0:
        # Doğal öneriler
        recommendations.extend([
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Doğal Kızarıklık Bakımı",
                brand="Doğal Çözüm",
                product_type="natural_remedy",
                description="Kızarıklık için doğal çözümler",
                price_range="low",
                ingredients="Aloe vera, Yeşil çay, Papatya, Yulaf ezmesi, Salatalık, Lavanta yağı, Hindistan cevizi yağı, Bal",
                usage_instructions="""Kullanım Bölgeleri: Yüz
1. Aloe vera: Taze aloe vera jeli kızarık bölgelere sürün, 15 dakika bekletin.
2. Yeşil çay toniği: Demlenmiş yeşil çayı soğutun, pamukla kızarık bölgelere uygulayın.
3. Papatya kompresi: Demlenmiş papatya çayını soğutun, pamukla kızarık bölgelere uygulayın.
4. Yulaf ezmesi maskesi: Yulaf ezmesini su ile karıştırın, kızarık bölgelere uygulayın.
5. Salatalık maskesi: Rendelenmiş salatalığı kızarık bölgelere uygulayın, 15 dakika bekletin.
6. Lavanta yağı: 1 damla lavanta yağını 1 yemek kaşığı hindistan cevizi yağı ile karıştırın, kızarık bölgelere uygulayın.
7. Hindistan cevizi yağı: Saf hindistan cevizi yağını kızarık bölgelere uygulayın.
8. Bal maskesi: Balı kızarık bölgelere sürün, 15 dakika bekletin."""
            )
        ])
        
        # Ürün önerileri
        recommendations.extend([
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Centella Asiatica Cream",
                brand="Dr. Jart+",
                product_type="moisturizer",
                description="Centella asiatica içeren krem, kızarıklığı azaltır ve cildi yatıştırır.",
                price_range="medium",
                ingredients="Centella Asiatica, Madecassoside, Asiaticoside",
                usage_instructions="Kullanım Bölgeleri: Yüz\nGünde 2 kez temiz cilde uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Green Tea Toner",
                brand="Isntree",
                product_type="toner",
                description="Yeşil çay özlü tonik, antioksidan etki gösterir ve kızarıklığı azaltır.",
                price_range="low",
                ingredients="Green Tea Extract, Hyaluronic Acid, Panthenol",
                usage_instructions="Kullanım Bölgeleri: Yüz\nTemizleme sonrası pamukla uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Azelaic Acid Treatment",
                brand="The Ordinary",
                product_type="treatment",
                description="Azelaik asit içeren tedavi ürünü, kızarıklığı azaltır ve cilt tonunu eşitler.",
                price_range="low",
                ingredients="Azelaic Acid, Hyaluronic Acid, Niacinamide",
                usage_instructions="Kullanım Bölgeleri: Yüz\nAkşamları temiz cilde uygulayın."
            ),
            ProductRecommendation(
                analysis_id=analysis.id,
                product_name="Calming Serum",
                brand="La Roche-Posay",
                product_type="serum",
                description="Yatıştırıcı serum, kızarıklığı azaltır ve cildi rahatlatır.",
                price_range="medium",
                ingredients="Niacinamide, Ceramides, Madecassoside",
                usage_instructions="Kullanım Bölgeleri: Yüz\nGünde 2 kez temiz cilde uygulayın."
            )
        ])
    
    # Genel bakım önerileri
    recommendations.append(
        ProductRecommendation(
            analysis_id=analysis.id,
            product_name="Doğal Günlük Bakım",
            brand="Doğal Çözüm",
            product_type="natural_remedy",
            description="Günlük cilt bakımı için doğal öneriler",
            price_range="low",
            ingredients="Su, Yeşil çay, Gül suyu, Hindistan cevizi yağı, Aloe vera, Bal, Zeytinyağı, Avokado",
            usage_instructions="""Kullanım Bölgeleri: Tüm yüz ve vücut
1. Günde en az 2 litre su için.
2. Yeşil çay tüketin (günde 2-3 fincan).
3. Gül suyu ile cildinizi ferahlatın.
4. Hindistan cevizi yağı ile cildinizi nemlendirin.
5. Aloe vera jeli ile cildinizi yatıştırın.
6. Bal maskesi ile cildinizi besleyin.
7. Zeytinyağı ile cildinizi nemlendirin.
8. Avokado maskesi ile cildinizi besleyin."""
        )
    )
    
    recommendations.append(
        ProductRecommendation(
            analysis_id=analysis.id,
            product_name="SPF 50+ Sunscreen",
            brand="La Roche-Posay",
            product_type="sunscreen",
            description="Yüksek korumalı güneş kremi, cildi UV hasarından korur.",
            price_range="medium",
            ingredients="Mexoryl XL, Titanium Dioxide, Glycerin",
            usage_instructions="Kullanım Bölgeleri: Tüm yüz ve vücut\nHer sabah dışarı çıkmadan önce uygulayın."
        )
    )
    
    return recommendations

@app.route('/delete_analysis/<int:analysis_id>', methods=['POST'])
def delete_analysis(analysis_id):
    if 'user_id' not in session:
        flash('Bu işlem için giriş yapmalısınız!')
        return redirect(url_for('login'))
    
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Kullanıcının kendi analizini sildiğinden emin oluyoruz
    if analysis.user_id != session['user_id']:
        flash('Bu analizi silme yetkiniz yok!')
        return redirect(url_for('history'))
    
    try:
        # Analiz görselini silme
        if analysis.image_path and os.path.exists(analysis.image_path):
            os.remove(analysis.image_path)
        
        # İlişkili ürün önerilerini silme
        ProductRecommendation.query.filter_by(analysis_id=analysis_id).delete()
        
        # Analiz kaydını silme
        db.session.delete(analysis)
        db.session.commit()
        
        flash('Analiz başarıyla silindi!')
    except Exception as e:
        db.session.rollback()
        flash('Analiz silinirken bir hata oluştu!')
        print(f"Analiz silme hatası: {str(e)}")
    
    return redirect(url_for('history'))

# Admin kontrolü için decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Bu sayfaya erişmek için giriş yapmalısınız.')
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash('Bu sayfaya erişim yetkiniz yok.')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Admin dashboard
@app.route('/admin')
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_analyses = Analysis.query.count()
    today_analyses = Analysis.query.filter(Analysis.date >= datetime.now().date()).count()
    total_recommendations = ProductRecommendation.query.count()
    
    # Son aktiviteleri al (son 10)
    recent_activities = []
    recent_analyses = Analysis.query.order_by(Analysis.date.desc()).limit(10).all()
    for analysis in recent_analyses:
        recent_activities.append({
            'date': analysis.date.strftime('%d/%m/%Y %H:%M'),
            'user': analysis.user.username,
            'action': 'Analiz Yaptı',
            'details': f'Cilt yaşı: {analysis.skin_age}'
        })
    
    return render_template('admin/dashboard.html',
                         total_users=total_users,
                         active_users=active_users,
                         total_analyses=total_analyses,
                         today_analyses=today_analyses,
                         total_recommendations=total_recommendations,
                         recent_activities=recent_activities)

# Kullanıcı yönetimi
@app.route('/admin/users')
@admin_required
def admin_users():
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('search', '')
    
    query = User.query
    if search_query:
        query = query.filter(
            (User.username.ilike(f'%{search_query}%')) |
            (User.email.ilike(f'%{search_query}%'))
        )
    
    users = query.order_by(User.created_at.desc()).paginate(page=page, per_page=10)
    return render_template('admin/users.html',
                         users=users.items,
                         current_page=page,
                         total_pages=users.pages,
                         search_query=search_query)

# Kullanıcı düzenleme
@app.route('/admin/users/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        try:
            user.username = request.form.get('username', user.username)
            user.email = request.form.get('email', user.email)
            user.age = request.form.get('age', type=int)
            user.gender = request.form.get('gender')
            user.skin_type = request.form.get('skin_type')
            user.skin_concerns = request.form.get('skin_concerns')
            user.allergies = request.form.get('allergies')
            user.is_active = 'is_active' in request.form
            user.is_admin = 'is_admin' in request.form
            
            db.session.commit()
            flash('Kullanıcı başarıyla güncellendi!')
            return redirect(url_for('admin_users'))
        except Exception as e:
            db.session.rollback()
            flash('Kullanıcı güncellenirken bir hata oluştu!')
            print(f"Kullanıcı güncelleme hatası: {str(e)}")
    
    return render_template('admin/edit_user.html', user=user)

# Kullanıcı silme
@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    try:
        # Kullanıcının analizlerini sil
        for analysis in user.analyses:
            if analysis.image_path and os.path.exists(analysis.image_path):
                os.remove(analysis.image_path)
            ProductRecommendation.query.filter_by(analysis_id=analysis.id).delete()
            db.session.delete(analysis)
        
        # Kullanıcıyı sil
        db.session.delete(user)
        db.session.commit()
        flash('Kullanıcı başarıyla silindi!')
    except Exception as e:
        db.session.rollback()
        flash('Kullanıcı silinirken bir hata oluştu!')
        print(f"Kullanıcı silme hatası: {str(e)}")
    
    return redirect(url_for('admin_users'))

# Analiz yönetimi
@app.route('/admin/analyses')
@admin_required
def admin_analyses():
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('search', '')
    date_filter = request.args.get('date', '')
    skin_type_filter = request.args.get('skin_type', '')
    
    query = Analysis.query.join(User)
    
    if search_query:
        query = query.filter(User.username.ilike(f'%{search_query}%'))
    
    if date_filter:
        query = query.filter(Analysis.date >= datetime.strptime(date_filter, '%Y-%m-%d'))
    
    if skin_type_filter:
        query = query.filter(Analysis.skin_type == skin_type_filter)
    
    analyses = query.order_by(Analysis.date.desc()).paginate(page=page, per_page=10)
    return render_template('admin/analyses.html',
                         analyses=analyses.items,
                         current_page=page,
                         total_pages=analyses.pages,
                         search_query=search_query,
                         date_filter=date_filter,
                         skin_type_filter=skin_type_filter)

# Analiz detayı
@app.route('/admin/analyses/<int:analysis_id>')
@admin_required
def admin_view_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    user = User.query.get(analysis.user_id)
    recommendations = ProductRecommendation.query.filter_by(analysis_id=analysis_id).all()
    
    return render_template('admin/view_analysis.html', 
                         analysis=analysis,
                         user=user,
                         recommendations=recommendations)

# Analiz silme
@app.route('/admin/analyses/<int:analysis_id>/delete', methods=['POST'])
@admin_required
def admin_delete_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    try:
        # Analiz görselini sil
        if analysis.image_path and os.path.exists(analysis.image_path):
            os.remove(analysis.image_path)
        
        # İlişkili ürün önerilerini silme
        ProductRecommendation.query.filter_by(analysis_id=analysis_id).delete()
        
        # Analiz kaydını silme
        db.session.delete(analysis)
        db.session.commit()
        
        flash('Analiz başarıyla silindi!')
    except Exception as e:
        db.session.rollback()
        flash('Analiz silinirken bir hata oluştu!')
        print(f"Analiz silme hatası: {str(e)}")
    
    return redirect(url_for('admin_analyses'))

# Ürün önerileri yönetimi
@app.route('/admin/recommendations')
@admin_required
def admin_recommendations():
    page = request.args.get('page', 1, type=int)
    
    recommendations = ProductRecommendation.query.order_by(ProductRecommendation.id.desc()).paginate(page=page, per_page=10)
    return render_template('admin/recommendations.html',
                         recommendations=recommendations.items,
                         current_page=page,
                         total_pages=recommendations.pages)

# Yeni ürün önerisi ekleme
@app.route('/admin/recommendations/add', methods=['POST'])
@admin_required
def admin_add_recommendation():
    try:
        # Form verilerini al
        product_name = request.form.get('product_name')
        brand = request.form.get('brand')
        product_type = request.form.get('product_type')
        price_range = request.form.get('price_range')
        description = request.form.get('description', '')
        ingredients = request.form.get('ingredients', '')
        usage_instructions = request.form.get('usage_instructions', '')
        target_skin_type = request.form.get('target_skin_type', '')
        
        # Zorunlu alanları kontrol et
        if not all([product_name, brand, product_type, price_range]):
            flash('Lütfen tüm zorunlu alanları doldurun!', 'error')
            return redirect(url_for('admin_recommendations'))
        
        # Yeni öneriyi oluştur
        recommendation = ProductRecommendation(
            product_name=product_name,
            brand=brand,
            product_type=product_type,
            price_range=price_range,
            description=description,
            ingredients=ingredients,
            usage_instructions=usage_instructions,
            target_skin_type=target_skin_type,
            is_active=True
        )
        
        db.session.add(recommendation)
        db.session.commit()
        flash('Ürün önerisi başarıyla eklendi!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Ürün önerisi eklenirken bir hata oluştu: {str(e)}', 'error')
        print(f"Ürün önerisi ekleme hatası: {str(e)}")
    
    return redirect(url_for('admin_recommendations'))

# Ürün önerisi düzenleme
@app.route('/admin/recommendations/<int:recommendation_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_recommendation(recommendation_id):
    recommendation = ProductRecommendation.query.get_or_404(recommendation_id)
    
    if request.method == 'POST':
        try:
            recommendation.product_name = request.form['product_name']
            recommendation.brand = request.form['brand']
            recommendation.product_type = request.form['product_type']
            recommendation.price_range = request.form['price_range']
            recommendation.description = request.form.get('description', '')
            recommendation.ingredients = request.form.get('ingredients', '')
            recommendation.usage_instructions = request.form.get('usage_instructions', '')
            recommendation.target_skin_type = request.form.get('target_skin_type', '')
            recommendation.is_active = 'is_active' in request.form
            
            db.session.commit()
            flash('Ürün önerisi başarıyla güncellendi!')
            return redirect(url_for('admin_recommendations'))
        except Exception as e:
            db.session.rollback()
            flash('Ürün önerisi güncellenirken bir hata oluştu!')
            print(f"Ürün önerisi güncelleme hatası: {str(e)}")
    
    return render_template('admin/edit_recommendation.html', recommendation=recommendation)

# Ürün önerisi silme
@app.route('/admin/recommendations/<int:recommendation_id>/delete', methods=['POST'])
@admin_required
def admin_delete_recommendation(recommendation_id):
    recommendation = ProductRecommendation.query.get_or_404(recommendation_id)
    
    try:
        db.session.delete(recommendation)
        db.session.commit()
        flash('Ürün önerisi başarıyla silindi!')
    except Exception as e:
        db.session.rollback()
        flash('Ürün önerisi silinirken bir hata oluştu!')
        print(f"Ürün önerisi silme hatası: {str(e)}")
    
    return redirect(url_for('admin_recommendations'))

@app.route('/live_support')
def live_support():
    return render_template('live_support.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    global chatbot
    try:
        # PDF dosyasının yolu
        pdf_path = os.path.join(os.path.dirname(__file__), 'chatbot', 'HocSkinBilgiler.pdf')
        
        # PDF dosyasının varlığını kontrol et
        if not os.path.exists(pdf_path):
            return jsonify({
                'status': 'error',
                'message': 'PDF dosyası bulunamadı. Lütfen sistem yöneticisiyle iletişime geçin.'
            })
        
        # Google API anahtarı
        api_key = "AIzaSyDqnJlj_dLF5jvUheUJ_BwZITb-bDICAj8"
        
        chatbot = PDFChatbot(api_key, pdf_path)
        
        return jsonify({
            'status': 'success',
            'message': 'Chatbot başarıyla başlatıldı!'
        })
    except Exception as e:
        print(f"Chatbot başlatma hatası: {str(e)}")  # Hata loglaması
        return jsonify({
            'status': 'error',
            'message': f'Hata oluştu: {str(e)}'
        })

@app.route('/ask', methods=['POST'])
def ask():
    global chatbot
    if not chatbot:
        return jsonify({
            'status': 'error',
            'message': 'Chatbot henüz başlatılmadı.'
        })
    
    try:
        question = request.json.get('question')
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'Lütfen bir soru girin.'
            })
        
        answer = chatbot.ask(question)
        
        # Eğer cevap "metinde bu yok" veya benzeri bir şey içeriyorsa
        if "metinde bu yok" in answer.lower() or "bulamadım" in answer.lower():
            alternative_response = """
            Üzgünüm, bu konu hakkında spesifik bir bilgim yok. Ancak size aşağıdaki konularda yardımcı olabilirim:

            • Cilt bakımı ve cilt sağlığı
            • Cilt analizi ve sonuçları
            • Cilt problemleri ve çözümleri
            • Ürün önerileri ve kullanımı
            • HocSkin hizmetleri

            Bu konulardan biriyle ilgili soru sorabilirsiniz. Size nasıl yardımcı olabilirim?
            """
            return jsonify({
                'status': 'success',
                'answer': alternative_response
            })
        
        return jsonify({
            'status': 'success',
            'answer': answer
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Hata oluştu: {str(e)}'
        })

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

if __name__ == '__main__':
    print("Template dizini:", template_dir)
    print("Mevcut çalışma dizini:", os.getcwd())
    app.run(host='0.0.0.0', port=5000, debug=True)

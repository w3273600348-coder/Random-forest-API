from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import requests
import xml.etree.ElementTree as ET
import re
import os

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')

GEMINI_KEY = os.environ.get('GEMINI_KEY', 'AIzaSyCqBZu6Hz6RN68iFdaiLGDkkfYfUQx--Qs')

ARXIV_CATEGORIES = ['cs.CL', 'cs.AI', 'cs.LG', 'stat.ML']

def fetch_arxiv_abstract():
    """从arXiv随机抓取一篇最新论文摘要"""
    import random
    cat = random.choice(ARXIV_CATEGORIES)
    url = f'http://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&max_results=20'
    try:
        r = requests.get(url, timeout=8)
        root = ET.fromstring(r.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        if not entries:
            return None, None, None
        entry = random.choice(entries[:10])
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        abstract = re.sub(r'\s+', ' ', abstract)
        # 截取前300词左右
        words = abstract.split()
        if len(words) > 120:
            abstract = ' '.join(words[:120]) + '...'
        return title, abstract, cat
    except Exception as e:
        return None, None, str(e)

def generate_gemini_text(title, abstract):
    """用Gemini生成同主题文本"""
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}'
    prompt = f"""Write a short academic paragraph (80-120 words) about the same topic as this paper, 
in a formal academic style. Paper title: "{title}". 
Write only the paragraph, no intro, no explanation."""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 200, "temperature": 0.7}
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        data = r.json()
        text = data['candidates'][0]['content']['parts'][0]['text'].strip()
        words = text.split()
        if len(words) > 120:
            text = ' '.join(words[:120]) + '...'
        return text
    except Exception as e:
        return None

def estimate_features(text):
    """从文本估算L2SCA特征（近似值）"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    n_sent = max(len(sentences), 1)
    words = text.split()
    n_words = len(words)
    
    # 估算各指标
    mls = n_words / n_sent
    
    # CN/T 估算：名词化词汇密度
    nominalizers = len(re.findall(r'\b\w+(tion|ment|ity|ness|ance|ence|ism|ogy|ure|age)\b', text, re.I))
    prepositions = len(re.findall(r'\b(of|in|for|with|on|at|by|from|through|between|among|within)\b', text, re.I))
    det_noun = len(re.findall(r'\b(the|a|an)\s+\w+\s+(of|in|for)\b', text, re.I))
    cnt_est = (nominalizers * 0.8 + prepositions * 0.15 + det_noun * 0.5) / n_sent
    cnt_est = max(0.8, min(4.0, cnt_est))
    
    # 从属子句估算
    subordinators = len(re.findall(r'\b(which|that|who|whom|whose|when|where|because|although|since|while|if|whether)\b', text, re.I))
    dc_t = subordinators / (n_sent * 1.2)
    
    # 其他特征估算
    vp_t = n_words / (n_sent * 6) + 1.5
    
    features = {
        'MLS': mls,
        'MLT': mls * 0.92,
        'MLC': mls * 0.55,
        'C_S': 1.75 + dc_t * 0.3,
        'C_T': 1.65 + dc_t * 0.25,
        'CT_T': min(0.9, 0.6 + dc_t * 0.4),
        'DC_C': min(0.6, dc_t * 0.6),
        'DC_T': min(0.9, dc_t),
        'CP_T': 0.5 + cnt_est * 0.05,
        'CP_C': 0.3 + cnt_est * 0.03,
        'CN_T': cnt_est,
        'CN_C': cnt_est * 0.6,
        'T_S': max(0.95, 1.15 - dc_t * 0.1),
        'VP_T': min(3.0, vp_t),
    }
    return features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = [float(data[f]) for f in feature_names]
        prob = model.predict_proba([features])[0]
        ai_prob = prob[1]
        label = 'AI-Generated' if ai_prob > 0.5 else 'Human-Authored'
        return jsonify({
            'label': label,
            'ai_probability': round(ai_prob * 100, 1),
            'human_probability': round(prob[0] * 100, 1),
            'confidence': round(max(prob) * 100, 1),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/live_detect', methods=['GET'])
def live_detect():
    """实时抓取arXiv摘要 + Gemini生成 + 双文本检测"""
    title, human_text, cat = fetch_arxiv_abstract()
    if not human_text:
        return jsonify({'status': 'error', 'message': 'arXiv fetch failed'})
    
    ai_text = generate_gemini_text(title, human_text)
    if not ai_text:
        return jsonify({'status': 'error', 'message': 'Gemini generation failed'})
    
    # 对两段文本做特征估算和预测
    def predict_text(text):
        feats = estimate_features(text)
        feat_vec = [feats[f] for f in feature_names]
        prob = model.predict_proba([feat_vec])[0]
        return {
            'ai_probability': round(prob[1] * 100, 1),
            'human_probability': round(prob[0] * 100, 1),
            'label': 'AI-Generated' if prob[1] > 0.5 else 'Human-Authored',
            'cn_t': round(feats['CN_T'], 3),
            'mls': round(feats['MLS'], 2),
        }
    
    human_result = predict_text(human_text)
    ai_result = predict_text(ai_text)
    
    return jsonify({
        'status': 'success',
        'title': title,
        'category': cat,
        'human_text': human_text,
        'ai_text': ai_text,
        'human_result': human_result,
        'ai_result': ai_result,
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

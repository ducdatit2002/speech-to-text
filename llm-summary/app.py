from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn
import os

app = FastAPI(title="Vietnamese LLM Summary API")

class TextInput(BaseModel):
    text: str

class FilePathInput(BaseModel):
    filepath: str

MODEL_NAME = "VietAI/vit5-large-vietnews-summarization"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
    print("Mô hình tóm tắt đã được tải thành công.")
except Exception as e:
    print(f"Lỗi khi tải mô hình tóm tắt: {e}")
    summarizer = None
    tokenizer = None

def split_text(text, max_tokens=512):
    if not tokenizer:
        raise ValueError("Tokenizer chưa được tải.")

    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        current_length = len(tokenizer.encode(current_chunk, add_special_tokens=False))
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
        
        if current_length + sentence_length + 1 > max_tokens:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            if sentence_length > max_tokens:
                # Chia nhỏ câu quá dài nếu cần
                words = sentence.split()
                temp_sentence = ""
                for word in words:
                    word_length = len(tokenizer.encode(temp_sentence + " " + word, add_special_tokens=False))
                    if word_length > max_tokens:
                        if temp_sentence.strip():
                            chunks.append(temp_sentence.strip())
                            temp_sentence = word
                        else:
                            print(f"Từ quá dài để xử lý: {word}")
                    else:
                        temp_sentence += " " + word
                if temp_sentence.strip():
                    chunks.append(temp_sentence.strip())
            else:
                current_chunk = sentence + ". "
        else:
            current_chunk += sentence + ". "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def summarize_text(text):
    if not summarizer:
        raise ValueError("Mô hình chưa sẵn sàng.")

    print("Bắt đầu quá trình tóm tắt...")
    chunks = split_text(text, max_tokens=512)
    print(f"Số đoạn sau khi chia: {len(chunks)}")

    # Đặt max_length rất lớn, ví dụ 2048, và min_length tương đối lớn để có bản tóm tắt dài
    max_len = 2048
    min_len = 1000  # Điều chỉnh tùy ý, nhưng phải nhỏ hơn max_len

    summaries = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Đang tóm tắt đoạn {i}/{len(chunks)} (độ dài: {len(chunk)} ký tự)")
        # Gọi summarizer với max_length rất lớn
        summary = summarizer(
            chunk, 
            max_length=max_len, 
            min_length=min_len, 
            do_sample=False
        )
        result_key = 'summary_text' if 'summary_text' in summary[0] else 'generated_text'
        summaries.append(summary[0][result_key])

    # Ghép tất cả tóm tắt thành một bản tóm tắt duy nhất
    final_summary = " ".join(summaries)

    print("Hoàn thành tóm tắt.")
    return final_summary


@app.post("/summarize")
def summarize_endpoint(input_data: TextInput):
    summary_result = summarize_text(input_data.text)
    return {"summary": summary_result}

@app.post("/summarize_file")
def summarize_file(input_data: FilePathInput):
    if not input_data.filepath:
        raise HTTPException(status_code=400, detail="File path không được để trống")
    
    if not os.path.exists(input_data.filepath):
        raise HTTPException(status_code=404, detail="File không tồn tại")

    try:
        with open(input_data.filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        summary = summarize_text(text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tóm tắt: {e}")

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

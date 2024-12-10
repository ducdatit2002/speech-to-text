### speech to text

# Vietnamese ASR with Wav2Vec2.0

Ứng dụng này cung cấp khả năng nhận dạng tiếng Việt từ file âm thanh sử dụng mô hình [Wav2Vec2.0](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) và language model n-gram.

## Thiết lập môi trường

1. Cài đặt [Miniconda](https://docs.conda.io/en/latest/miniconda.html) hoặc Anaconda (nếu chưa có).

2. Tạo môi trường Conda và kích hoạt:

   ```bash
   conda create -n asr python=3.11 -y
   conda activate asr
Cài đặt các thư viện cần thiết qua pip:
```bash
pip install -r requirements.txt
```
```bash
pip install streamlit transformers soundfile librosa numpy pyctcdecode kenlm huggingface_hub
```
```bash
python main.py --audio demo1h.mp3
```
```bash
streamlit run app.py
```
Mở trình duyệt web và truy cập link được cung cấp (thường là http://localhost:8501).

3. Ghi chú
Thư mục ./cache sẽ được sử dụng để lưu trữ model và tokenizer.
Thư mục ./uploads sẽ được tạo ra và bỏ qua trong git (xem .gitignore). Audio tải lên sẽ được lưu tại đó.

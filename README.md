---
title: RAG Chatbot — Streamlit
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---


# 📝 Smart Resume Optimizer

The **Smart Resume Optimizer** is an AI-powered tool that tailors resumes to specific job descriptions.  
It takes a user’s original resume (PDF or LaTeX), a job description, and optional prompts, then generates a **customized resume** optimized for the target role.  

This project is designed for job seekers, data scientists, and ML engineers who want to stand out with targeted applications while also serving as a portfolio project showcasing **MLOps, FastAPI, Docker, CI/CD, and AI engineering**.

---

## 🚀 Features

- 📄 **Resume Uploads**  
  - Accepts **PDF** resumes and original **LaTeX** source files.
  - Preserves LaTeX formatting when tailoring LaTeX resumes.

- 🤖 **LLM-powered Resume Optimization**  
  - Extracts skills and experiences from the original resume.  
  - Matches relevant content to the target job description.  
  - Outputs an **optimized PDF resume**.

- 🎯 **Customization**  
  - Add custom prompts (e.g., “highlight leadership skills” or “emphasize bioinformatics projects”).  
  - Condensed and ATS-friendly outputs.

- 🛠️ **Tech Stack**  
  - **Backend**: FastAPI (Python)  
  - **Frontend**: React + TailwindCSS  
  - **AI**: OpenAI / OpenRouter API (configurable)  
  - **Resume Rendering**: Pandoc + LaTeX templates  
  - **Containerization**: Docker + GitHub Codespaces  
  - **Deployment**: Render (backend), Vercel (frontend), optional Hugging Face Spaces

---

## 🏗️ Architecture

```
User → React Frontend → FastAPI Backend → LLM API → Pandoc/LaTeX → Optimized Resume (PDF)
```

- **Frontend (Vercel)**: Clean UI for uploading resumes, entering job descriptions, and downloading results.  
- **Backend (Render / Docker)**: FastAPI server handles resume processing, calls LLM API, and manages LaTeX rendering.  
- **Pandoc/LaTeX**: Ensures professional PDF formatting.  
- **GitHub Actions CI/CD**: Automated testing, build, and deployment pipelines.  

---

## ⚙️ Installation & Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/smart-resume-optimizer.git
cd smart-resume-optimizer
```

### 2. Backend setup (FastAPI)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run the server:
```bash
uvicorn app:app --reload --port 8000
```
API available at: [http://localhost:8000](http://localhost:8000)

### 3. Frontend setup (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
UI available at: [http://localhost:5173](http://localhost:5173)

### 4. Environment variables
Create `.env` in `backend/`:
```ini
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
```

### 5. Docker (optional)
To build and run the backend with Docker:
```bash
docker build -t smart-resume-backend ./backend
docker run -p 8000:8000 smart-resume-backend
```

---

## ⚡ Usage

1. Upload your resume (**PDF** or **LaTeX**).  
2. Paste the **job description** or link.  
3. (Optional) Add extra instructions.  
4. Click **Generate Optimized Resume**.  
5. Download the tailored PDF.  

---

## 🔧 Configuration

- `config.yaml` (optional) to customize defaults:  
  ```yaml
  model: gpt-4o-mini
  output_format: pdf
  preserve_sections: true
  ```
- `.env` to set API keys and deployment configs.  
- Custom **LaTeX templates** supported via `templates/`.

---

## 📦 Deployment

### Render (Backend)
- Push backend Docker image to **AWS ECR** or **DockerHub**.  
- Deploy FastAPI app on **Render** (or Hugging Face Spaces).  

### Vercel (Frontend)
- Deploy frontend directly via `vercel deploy`.  
- Set `NEXT_PUBLIC_API_URL` to backend endpoint.  

### GitHub Actions (CI/CD)
- Automated build & deploy workflow in `.github/workflows/deploy.yml`.

---

## 📚 Example

Input resume (PDF):  
```
Skills: Python, Machine Learning, Bioinformatics
Projects: scRNA-seq cell typing, Resume Optimizer
```

Job description:  
```
Looking for a Machine Learning Engineer with experience in NLP, MLOps, and FastAPI.
```

Optimized resume output:  
- Highlights **NLP & MLOps** experience.  
- Moves **FastAPI projects** to top of “Projects” section.  
- Keeps LaTeX formatting intact.  

---

## 🤝 Contributing

Pull requests are welcome!  
1. Fork the repo  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m 'Added X feature'`)  
4. Push (`git push origin feature-name`)  
5. Open a Pull Request  

---

## 📄 License

This project is licensed under the MIT License.  
See `LICENSE` for details.  

---

## 👨‍💻 Author

**Mostafa Malmir**  
- PhD Candidate, AI/ML Engineer  
- [GitHub](https://github.com/MoMalmir) | [LinkedIn](https://linkedin.com/in/mostafa-malmir)  

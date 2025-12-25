# 1. Start with a base Linux system that has Python 3.9 installed
FROM python:3.9-slim

# 2. Create a folder inside the box for our app
WORKDIR /app

# 3. Copy the "requirements.txt" file into the box
COPY requirements.txt .

# 4. Install the libraries inside the box
# --no-cache-dir keeps the box small
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn

# 5. Copy all our code and models into the box
COPY . .

# 6. Open a hole in the box (Port 8000) so we can talk to the app
EXPOSE 8000

# 7. Tell the box what to do when it turns on
# "Run uvicorn server on port 8000"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
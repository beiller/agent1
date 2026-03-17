FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git curl wget net-tools iputils-ping jq tmux grep sed gawk gcc procps

# Copy application code
COPY . .

# Run the async main function
CMD ["python", "main.py"]

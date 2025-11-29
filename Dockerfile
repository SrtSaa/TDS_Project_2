FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# 1. Install Python packages (including playwright)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 2. Install System Dependencies (Must be ROOT)
# This installs Ubuntu libraries (libnss3, libatk, etc.) required to RUN the browser
RUN playwright install-deps chromium

# 3. Create the user
RUN useradd -m -u 1000 user

# 4. Switch to User
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# 5. Install Browser Binaries (Must be USER)
# This downloads the actual Chrome/Chromium binary to /home/user/.cache/ms-playwright
RUN playwright install chromium

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
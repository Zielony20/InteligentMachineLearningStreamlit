FROM python:3.7

ADD requirements.txt .
ADD start.py .
ADD loadInterface.py .
ADD Functions/* ./Functions/
ADD widget.json .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip install streamlit --upgrade --no-cache-dir

RUN streamlit run start.py

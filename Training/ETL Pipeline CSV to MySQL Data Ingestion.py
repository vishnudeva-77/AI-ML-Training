#!/usr/bin/env python
# coding: utf-8

# In[4]:


import logging
import pandas as pd
import numpy as np
import mysql.connector
import datetime
import os

# Set up basic logging
logging.basicConfig(
    filename='etl_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# In[5]:


csv_file_path = 'employees1.csv'  # Make sure this file is in your working directory

df = pd.read_csv(csv_file_path)

print("Raw data loaded:")
print(df.head())
print("Column names:", df.columns.tolist())

logging.info("CSV loaded successfully.")


# In[6]:


# Fill missing values
df.fillna({
    'EMAIL': 'not_provided@example.com',
    'PHONE_NUMBER': '0000000000',
    'HIRE_DATE': '01-Jan-00',
    'SALARY': 0
}, inplace=True)

# Standardize column names
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
print("Cleaned column names:", df.columns.tolist())


# In[7]:


# Convert 'hire_date' to datetime
df['hire_date'] = pd.to_datetime(df['hire_date'], format='%d-%b-%y', errors='coerce')

# Replace invalid dates with default
df['hire_date'] = df['hire_date'].fillna(pd.to_datetime('2000-01-01'))

# Convert 'salary' to numeric
df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(0).astype(int)

logging.info("Data cleaning completed.")


# In[8]:


# Create connection without database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Aditiyak@3821"
)

cursor = mydb.cursor()

# Create the database if it doesn't exist
cursor.execute("CREATE DATABASE IF NOT EXISTS employee")

# ✅ Reconnect with the selected database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Aditiyak@3821",
    database="employee"
)
cursor = mydb.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS salary_2 (
        empid INT PRIMARY KEY,
        firstname VARCHAR(50),
        lastname VARCHAR(50),
        email VARCHAR(100),
        phone VARCHAR(20),
        hire_date DATE,
        job_id VARCHAR(20),
        salary INT
    )
""")


# In[9]:


for index, row in df.iterrows():
    sql = """
        INSERT INTO salary_2 (
            empid, firstname, lastname, email, phone, hire_date, job_id, salary
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            firstname=VALUES(firstname),
            lastname=VALUES(lastname),
            email=VALUES(email),
            phone=VALUES(phone),
            hire_date=VALUES(hire_date),
            job_id=VALUES(job_id),
            salary=VALUES(salary)
    """

    values = (
        int(row['employee_id']),
        row['first_name'],
        row['last_name'],
        row['email'],
        row['phone_number'],
        row['hire_date'].date(),
        row['job_id'],
        int(row['salary'])
    )

    cursor.execute(sql, values)

mydb.commit()
cursor.close()
mydb.close()

logging.info("ETL process completed successfully.")
print("ETL process completed successfully.")


# In[10]:


print("Current Working Directory:", os.getcwd())


# In[11]:


get_ipython().system('jupyter nbconvert --to script ETL Pipeline CSV to MySQL Data Ingestion.ipynb')


# In[ ]:





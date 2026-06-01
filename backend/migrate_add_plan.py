import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, database='postgres', user='postgres', password='admin123')
cur = conn.cursor()
for sql in [
    'ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS plan_expires_at TIMESTAMP;',
    'ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS daily_limit INTEGER DEFAULT 10;'
]:
    try:
        cur.execute(sql)
        print('OK:', sql[:60])
    except Exception as e:
        print('Skip:', e)
conn.commit()
cur.close()
conn.close()
print('Done!')

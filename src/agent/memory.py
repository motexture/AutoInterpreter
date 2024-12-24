import sqlite3
import os

class Memory:
    def __init__(self, path: str):
        self.path = path
        self._initialize_db()

    def _initialize_db(self):
        try:
            os.remove(self.path)
        except:
            pass

        self.conn = sqlite3.connect(self.path)
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS operations
                          (id INTEGER PRIMARY KEY, text TEXT)''')
        self.conn.commit()

    def memorize(self, text: str):
        self.c.execute("INSERT INTO operations (text) VALUES (?)", (text,))
        self.conn.commit()

    def remember(self, memories: int = 5) -> str:
        query = "SELECT id, text FROM operations ORDER BY id DESC LIMIT ?"
        self.c.execute(query, (memories,))
        
        rows = self.c.fetchall()
        rows.sort(key=lambda x: x[0], reverse=True)

        return '\n\n'.join(row[1].strip() for row in rows)
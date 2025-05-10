import sqlite3

def add_notes_column():
    try:
        # Connect to the database
        conn = sqlite3.connect('student_dropout.db')
        cursor = conn.cursor()
        
        # Add notes column
        cursor.execute('ALTER TABLE student ADD COLUMN notes TEXT')
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print("Successfully added notes column to student table")
        
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Notes column already exists")
        else:
            print(f"Error adding notes column: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == '__main__':
    add_notes_column() 
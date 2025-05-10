import sqlite3

def create_usertable():
    # Create a new database connection within the current thread
    conn = sqlite3.connect("usersdata.db")
    
    # Create a cursor for this thread
    c = conn.cursor()

    # Define and execute the SQL command to create the user table
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Call the function to create the user table
create_usertable()

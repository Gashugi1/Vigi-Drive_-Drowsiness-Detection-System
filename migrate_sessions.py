"""
Database migration script to add monitoring_session table

Run this script to update the database schema:
    python migrate_sessions.py
"""

from app import app, db

def migrate():
    with app.app_context():
        print("[MIGRATION] Creating monitoring_session table...")
        db.create_all()
        print("[MIGRATION] ✅ Migration complete!")
        
        # Verify table was created
        from app import MonitoringSession
        count = MonitoringSession.query.count()
        print(f"[MIGRATION] monitoring_session table exists. Current sessions: {count}")

if __name__ == "__main__":
    migrate()

from app import app, db, DrowsinessEvent
import time

with app.app_context():
    print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
    try:
        event = DrowsinessEvent(
            user_id=2,
            ear=0.15,
            mar=0.0,
            p_drowsy=0.95,
            state="test_event_user_2"
        )
        db.session.add(event)
        db.session.commit()
        print("Successfully inserted test event.")
        
        # Verify
        count = DrowsinessEvent.query.count()
        print(f"Total events in DB: {count}")
        
    except Exception as e:
        print(f"Error: {e}")

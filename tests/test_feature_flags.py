import unittest
import json
from app import app

class TestFeatureFlags(unittest.TestCase):
    """Tests for feature flag system"""
    
    def setUp(self):
        """Set up test client"""
        app.config['TESTING'] = True
        self.client = app.test_client()
    
    def test_api_features_endpoint(self):
        """Test that /api/features returns feature flags"""
        response = self.client.get('/api/features')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'application/json')
        
        data = response.get_json()
        self.assertIn('sound_alerts', data)
        self.assertIsInstance(data['sound_alerts'], dict)
    
    def test_sound_alerts_config(self):
        """Test that sound_alerts has required fields"""
        response = self.client.get('/api/features')
        data = response.get_json()
        
        sound_config = data.get('sound_alerts', {})
        self.assertIn('enabled', sound_config)
        self.assertIn('sound_file', sound_config)
        self.assertIsInstance(sound_config['enabled'], bool)
        self.assertIsInstance(sound_config['sound_file'], str)
    
    def test_feature_flags_from_config(self):
        """Test that feature flags are loaded from config.json"""
        # This test verifies the config.json file is valid
        try:
            with open('config.json', 'r') as f:
                cfg = json.load(f)
                self.assertIn('features', cfg)
                self.assertIn('sound_alerts', cfg['features'])
        except FileNotFoundError:
            self.fail("config.json not found")
        except json.JSONDecodeError:
            self.fail("config.json is not valid JSON")

if __name__ == '__main__':
    unittest.main()

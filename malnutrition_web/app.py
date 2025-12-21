from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

app = Flask(__name__)

class MalnutritionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_metrics = {}
        self.feature_importance = {}
        self.load_or_train_model()
    
    def generate_who_based_data(self, n_samples=3000):
        """Generate data berdasarkan WHO Growth Standards dengan Gender & MUAC"""
        np.random.seed(42)
        
        # WHO Standards: age -> (weight_male, height_male, weight_female, height_female, muac)
        who_standards = {
            0: (3.3, 49.9, 3.2, 49.1, 11.0),
            6: (7.9, 67.6, 7.3, 65.7, 13.3),
            12: (9.6, 75.7, 8.9, 74.0, 14.2),
            24: (12.2, 87.1, 11.5, 85.7, 15.0),
            36: (14.3, 96.1, 13.9, 95.1, 15.5),
            48: (16.3, 103.3, 15.9, 102.7, 16.0),
            60: (18.3, 109.9, 17.9, 108.4, 16.5)
        }
        
        data = []
        labels = []
        
        proportions = {
            'normal': 0.65,
            'stunted': 0.15,
            'wasted': 0.10,
            'overweight': 0.10
        }
        
        for i in range(n_samples):
            age = np.random.uniform(0, 60)
            gender = np.random.choice([0, 1])  # 0=female, 1=male
            
            ref_ages = sorted(who_standards.keys())
            closest_age = min(ref_ages, key=lambda x: abs(x - age))
            weight_m, height_m, weight_f, height_f, muac_ref = who_standards[closest_age]
            
            age_factor = age / 60
            
            if gender == 1:  # Male
                med_weight = weight_m + (18.3 - weight_m) * age_factor
                med_height = height_m + (109.9 - height_m) * age_factor
            else:  # Female
                med_weight = weight_f + (17.9 - weight_f) * age_factor
                med_height = height_f + (108.4 - height_f) * age_factor
            
            med_muac = 11.0 + (16.5 - 11.0) * age_factor
            
            weight_sd = 0.5 + (2.3 - 0.5) * age_factor
            height_sd = 2.0 + (5.0 - 2.0) * age_factor
            muac_sd = 0.8 + (1.2 - 0.8) * age_factor
            
            rand = np.random.random()
            
            if rand < proportions['normal']:
                weight_z = np.random.uniform(-0.8, 0.8)
                height_z = np.random.uniform(-0.8, 0.8)
                muac_z = np.random.uniform(-0.5, 0.8)
                weight = med_weight + (weight_z * weight_sd)
                height = med_height + (height_z * height_sd)
                muac = med_muac + (muac_z * muac_sd)
                label = 0
                
            elif rand < proportions['normal'] + proportions['stunted']:
                weight_z = np.random.uniform(-1.5, 0)
                height_z = np.random.uniform(-3.5, -2.0)
                muac_z = np.random.uniform(-1.0, 0)
                weight = med_weight + (weight_z * weight_sd)
                height = med_height + (height_z * height_sd)
                muac = med_muac + (muac_z * muac_sd)
                label = 1
                
            elif rand < proportions['normal'] + proportions['stunted'] + proportions['wasted']:
                weight_z = np.random.uniform(-3.5, -2.0)
                height_z = np.random.uniform(-1.0, 1.0)
                muac_z = np.random.uniform(-3.0, -1.5)
                weight = med_weight + (weight_z * weight_sd)
                height = med_height + (height_z * height_sd)
                muac = med_muac + (muac_z * muac_sd)
                label = 2
                
            else:
                weight_z = np.random.uniform(2.0, 3.5)
                height_z = np.random.uniform(-0.5, 1.0)
                muac_z = np.random.uniform(1.5, 2.5)
                weight = med_weight + (weight_z * weight_sd)
                height = med_height + (height_z * height_sd)
                muac = med_muac + (muac_z * muac_sd)
                label = 3
            
            weight = max(2.0, weight)
            height = max(40.0, height)
            muac = max(8.0, min(20.0, muac))
            
            bmi = weight / ((height/100) ** 2)
            weight_for_age_z = (weight - med_weight) / weight_sd
            height_for_age_z = (height - med_height) / height_sd
            muac_z_score = (muac - med_muac) / muac_sd
            
            data.append([age, gender, weight, height, muac, bmi, weight_for_age_z, height_for_age_z, muac_z_score])
            labels.append(label)
        
        return np.array(data), np.array(labels)
    
    def train_model(self):
        """Train Random Forest model with cross-validation"""
        print("🔄 Training Random Forest Model with Gender & MUAC...")
        print("=" * 60)
        
        X, y = self.generate_who_based_data(n_samples=3000)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaling features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # RANDOM FOREST CLASSIFIER dengan hyperparameter optimal
        print("🌲 Initializing Random Forest Classifier...")
        print(f"   - Number of Trees: 200")
        print(f"   - Max Depth: 20")
        print(f"   - Min Samples Split: 4")
        print(f"   - Min Samples Leaf: 2")
        print(f"   - Class Weight: Balanced")
        print(f"   - Criterion: Gini")
        
        self.model = RandomForestClassifier(
            n_estimators=200,          # Jumlah decision trees dalam forest
            max_depth=20,              # Kedalaman maksimal setiap tree
            min_samples_split=4,       # Min sampel untuk split node
            min_samples_leaf=2,        # Min sampel di leaf node
            max_features='sqrt',       # Fitur untuk split (sqrt dari total features)
            bootstrap=True,            # Bootstrap sampling
            oob_score=True,            # Out-of-bag score untuk validasi
            random_state=42,
            class_weight='balanced',   # Handle imbalanced classes
            n_jobs=-1                  # Gunakan semua CPU cores
        )
        
        print("\n🔧 Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score (5-fold)
        print("\n📊 Performing 5-Fold Cross-Validation...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Feature importance
        feature_names = ['Age', 'Gender', 'Weight', 'Height', 'MUAC', 'BMI', 'Weight_Z', 'Height_Z', 'MUAC_Z']
        importances = self.model.feature_importances_
        feature_importance_dict = dict(zip(feature_names, importances))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🎯 Feature Importance (Random Forest):")
        for feature, importance in sorted_features:
            print(f"   {feature:12s}: {importance:.4f} {'█' * int(importance * 50)}")
        
        # Out-of-bag score
        oob_score = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        
        self.model_metrics = {
            'accuracy': accuracy,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'oob_score': oob_score,
            'classification_report': classification_report(y_test, y_pred, 
                target_names=['Normal', 'Stunted', 'Wasted', 'Overweight']),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance_dict,
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_trees': 200,
            'max_depth': 20
        }
        
        print("\n" + "=" * 60)
        print(f"✅ Model Training Complete!")
        print(f"   Test Accuracy: {accuracy*100:.2f}%")
        print(f"   CV Mean Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
        if oob_score:
            print(f"   OOB Score: {oob_score*100:.2f}%")
        print("=" * 60)
        
        self.save_model()
    
    def save_model(self):
        """Save model"""
        try:
            with open('malnutrition_model_v2.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'metrics': self.model_metrics
                }, f)
            print("💾 Model saved successfully!")
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
    
    def load_model(self):
        """Load model"""
        try:
            if os.path.exists('malnutrition_model_v2.pkl'):
                with open('malnutrition_model_v2.pkl', 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.scaler = saved_data['scaler']
                    self.model_metrics = saved_data['metrics']
                print("✅ Random Forest Model loaded successfully!")
                print(f"   - Trees: {self.model.n_estimators}")
                print(f"   - Accuracy: {self.model_metrics['accuracy']*100:.2f}%")
                return True
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
        return False
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        if not self.load_model():
            self.train_model()
    
    def calculate_z_score(self, age, gender, weight, height, muac):
        """Calculate Z-scores with gender consideration"""
        age_factor = min(age / 60, 1.0)
        
        if gender == 1:  # Male
            expected_weight = 3.3 + (18.3 - 3.3) * age_factor
            expected_height = 49.9 + (109.9 - 49.9) * age_factor
        else:  # Female
            expected_weight = 3.2 + (17.9 - 3.2) * age_factor
            expected_height = 49.1 + (108.4 - 49.1) * age_factor
        
        expected_muac = 11.0 + (16.5 - 11.0) * age_factor
        
        weight_sd = 0.5 + (2.3 - 0.5) * age_factor
        height_sd = 2.0 + (5.0 - 2.0) * age_factor
        muac_sd = 0.8 + (1.2 - 0.8) * age_factor
        
        weight_z = (weight - expected_weight) / weight_sd
        height_z = (height - expected_height) / height_sd
        muac_z = (muac - expected_muac) / muac_sd
        
        return weight_z, height_z, muac_z
    
    def get_muac_status(self, muac, age):
        """Get MUAC nutritional status"""
        if age >= 6:
            if muac < 11.5:
                return "SAM (Severe Acute Malnutrition)", "danger"
            elif muac < 12.5:
                return "MAM (Moderate Acute Malnutrition)", "warning"
            else:
                return "Normal", "success"
        return "N/A (< 6 bulan)", "info"
    
    def predict(self, age, gender, weight, height, muac):
        """Make prediction using Random Forest"""
        bmi = weight / ((height/100) ** 2)
        weight_z, height_z, muac_z = self.calculate_z_score(age, gender, weight, height, muac)
        
        input_data = np.array([[age, gender, weight, height, muac, bmi, weight_z, height_z, muac_z]])
        input_scaled = self.scaler.transform(input_data)
        
        # Random Forest prediction
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        # Get decision path info (optional - untuk debugging)
        n_nodes = self.model.estimators_[0].tree_.node_count
        
        status_map = {
            0: ("NORMAL", "Anak memiliki status gizi yang baik", "success"),
            1: ("STUNTED (Pendek)", "Anak mengalami kekurangan gizi kronis", "warning"),
            2: ("WASTED (Kurus)", "Anak mengalami kekurangan gizi akut", "danger"),
            3: ("OVERWEIGHT (Kelebihan Berat)", "Anak memiliki berat badan berlebih", "info")
        }
        
        status, description, alert_type = status_map[prediction]
        muac_status, muac_alert = self.get_muac_status(muac, age)
        
        return {
            'status': status,
            'description': description,
            'alert_type': alert_type,
            'confidence': float(probabilities[prediction] * 100),
            'probabilities': {
                'Normal': float(probabilities[0] * 100),
                'Stunted': float(probabilities[1] * 100),
                'Wasted': float(probabilities[2] * 100),
                'Overweight': float(probabilities[3] * 100)
            },
            'bmi': float(bmi),
            'weight_z': float(weight_z),
            'height_z': float(height_z),
            'muac_z': float(muac_z),
            'muac_status': muac_status,
            'muac_alert': muac_alert,
            'prediction': int(prediction),
            'model_info': {
                'algorithm': 'Random Forest Classifier',
                'n_trees': self.model.n_estimators,
                'max_depth': self.model.max_depth
            }
        }

# Initialize model
ml_model = MalnutritionModel()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', accuracy=ml_model.model_metrics['accuracy']*100)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        
        age = float(data['age'])
        gender = int(data['gender'])
        weight = float(data['weight'])
        height = float(data['height'])
        muac = float(data['muac'])
        
        # Validasi
        if age < 0 or age > 60:
            return jsonify({'error': 'Umur harus antara 0-60 bulan!'}), 400
        if gender not in [0, 1]:
            return jsonify({'error': 'Gender tidak valid!'}), 400
        if weight <= 0 or weight > 30:
            return jsonify({'error': 'Berat badan tidak valid!'}), 400
        if height <= 0 or height > 150:
            return jsonify({'error': 'Tinggi badan tidak valid!'}), 400
        if muac <= 0 or muac > 25:
            return jsonify({'error': 'MUAC tidak valid!'}), 400
        
        result = ml_model.predict(age, gender, weight, height, muac)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Model metrics page"""
    return render_template('metrics.html', metrics=ml_model.model_metrics)

@app.route('/feature-importance')
def feature_importance():
    """Feature importance endpoint"""
    if 'feature_importance' in ml_model.model_metrics:
        return jsonify(ml_model.model_metrics['feature_importance'])
    return jsonify({'error': 'Feature importance not available'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
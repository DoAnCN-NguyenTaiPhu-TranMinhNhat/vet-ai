#!/usr/bin/env python3
"""
Test script to verify prediction-feedback ID consistency
"""

import asyncio
import sys
import os

# Add the ai_service directory to the path
sys.path.append('/home/teifu142/ATE/UIT/DACN/vet-ai/ai_service')

from continuous_training import PredictionLog, log_prediction, save_feedback, prediction_logs, feedback_data
from datetime import datetime

async def test_id_consistency():
    """Test that prediction and feedback IDs match"""
    print("=== Testing Prediction-Feedback ID Consistency ===\n")
    
    # Clear existing data
    prediction_logs.clear()
    feedback_data.clear()
    
    print("1. Testing prediction logging with sequential IDs...")
    
    # Test logging predictions with different IDs
    test_predictions = [
        {
            "id": 1,
            "visit_id": 10,
            "pet_id": 5,
            "prediction_input": {"symptom": "cough"},
            "prediction_output": {"diagnosis": "Respiratory Infection"},
            "model_version": "v2.0",
            "confidence_score": 0.85,
            "top_k_predictions": [{"diagnosis": "Respiratory Infection", "confidence": 0.85}],
            "veterinarian_id": 1,
            "clinic_id": 1
        },
        {
            "id": 2,
            "visit_id": 11,
            "pet_id": 6,
            "prediction_input": {"symptom": "vomiting"},
            "prediction_output": {"diagnosis": "Gastroenteritis"},
            "model_version": "v2.0",
            "confidence_score": 0.92,
            "top_k_predictions": [{"diagnosis": "Gastroenteritis", "confidence": 0.92}],
            "veterinarian_id": 2,
            "clinic_id": 1
        }
    ]
    
    # Log predictions
    for pred_data in test_predictions:
        prediction = PredictionLog(**pred_data)
        logged_id = await log_prediction(prediction)
        print(f"   Logged prediction with ID: {logged_id}")
    
    print(f"\n   Total predictions logged: {len(prediction_logs)}")
    
    print("\n2. Testing feedback with matching IDs...")
    
    # Test feedback with matching IDs
    test_feedback = [
        {
            "prediction_id": 1,
            "final_diagnosis": "Respiratory Infection",
            "is_correct": True,
            "confidence_rating": 5,
            "comments": "Perfect match",
            "veterinarian_id": 1,
            "is_training_eligible": True,
            "data_quality_score": 1.0
        },
        {
            "prediction_id": 2,
            "final_diagnosis": "Gastroenteritis",
            "is_correct": True,
            "confidence_rating": 4,
            "comments": "Good prediction",
            "veterinarian_id": 2,
            "is_training_eligible": True,
            "data_quality_score": 0.9
        }
    ]
    
    # Save feedback
    from continuous_training import DoctorFeedback
    for feedback_data in test_feedback:
        feedback = DoctorFeedback(**feedback_data)
        success = await save_feedback(feedback)
        print(f"   Saved feedback for prediction ID: {feedback.prediction_id}, Success: {success}")
    
    print(f"\n   Total feedback saved: {len(feedback_data)}")
    
    print("\n3. Verifying ID consistency...")
    
    # Check that all feedback prediction_ids exist in predictions
    feedback_ids = {f["prediction_id"] for f in feedback_data}
    prediction_ids = {p["id"] for p in prediction_logs}
    
    print(f"   Prediction IDs in storage: {sorted(prediction_ids)}")
    print(f"   Feedback prediction IDs: {sorted(feedback_ids)}")
    
    missing_predictions = feedback_ids - prediction_ids
    if missing_predictions:
        print(f"   ❌ ERROR: Feedback references missing predictions: {missing_predictions}")
        return False
    else:
        print(f"   ✅ All feedback IDs match prediction IDs")
    
    print("\n4. Testing training engine lookup...")
    
    # Test training engine lookup logic
    from training_engine import collect_training_data
    
    try:
        training_records = collect_training_data(feedback_data, prediction_logs)
        print(f"   ✅ Training engine found {len(training_records)} valid records")
        
        for i, record in enumerate(training_records):
            print(f"      Record {i+1}: prediction_id={record['prediction_id']}, features={len(record['features'])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training engine failed: {e}")
        return False

async def main():
    """Main test function"""
    try:
        success = await test_id_consistency()
        if success:
            print("\n🎉 All tests passed! ID consistency is working correctly.")
        else:
            print("\n❌ Tests failed! ID consistency issues detected.")
        return success
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(main())

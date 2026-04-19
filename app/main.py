# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # import joblib
# # import numpy as np
# # import os
# # from fastapi.middleware.cors import CORSMiddleware
# # from google import genai


# # # 🔴 TEMP ONLY — do NOT commit this
# # GEMINI_API_KEY = "AIzaSyCcbvEV7j0pXVQRhjfsNiefnxaxRAdSDPE"

# # client = None
# # try:
# #     client = genai.Client(api_key=GEMINI_API_KEY)
# #     print("Gemini initialized (hardcoded key) ✅")
# # except Exception as e:
# #     print("Gemini init error:", e)

# # app = FastAPI(title="Financial Fraud Detection API")

# # # ---------------- GEMINI SETUP ---------------- #
# # # We use .strip() to remove any accidental spaces or newlines from the terminal
# # # RAW_KEY = os.getenv("GEMINI_API_KEY")
# # # if RAW_KEY:
# # #     GEMINI_API_KEY = RAW_KEY.strip().strip('"').strip("'")
# # # else:
# # #     GEMINI_API_KEY = None

# # # client = None

# # if GEMINI_API_KEY:
# #     try:
# #         # Initialize the new Google GenAI client
# #         client = genai.Client(api_key=GEMINI_API_KEY)
# #         # Verify the key is being read (masked)
# #         print(
# #             f"Gemini initialized with key: {GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]} ✅"
# #         )
# #     except Exception as e:
# #         print(f"Gemini initialization error: {e}")
# #         client = None
# # else:
# #     print("⚠️ GEMINI_API_KEY not found in environment variables.")

# # # ---------------- CORS ---------------- #
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )


# # # ---------------- INPUT MODEL ---------------- #
# # class Transaction(BaseModel):
# #     step: int
# #     type: str
# #     amount: float


# # # ---------------- LOAD ML MODEL ---------------- #
# # model_data = None


# # @app.on_event("startup")
# # def load_model():
# #     global model_data
# #     model_path = "model/fraud_model.pkl"
# #     if os.path.exists(model_path):
# #         model_data = joblib.load(model_path)
# #         print("ML Model loaded successfully.")
# #     else:
# #         print("❌ ML Model not found. Please run train.py first.")


# # # ---------------- ROUTES ---------------- #
# # @app.get("/")
# # def home():
# #     return {"message": "Fraud Detection API is running. Use /predict"}


# # @app.post("/predict")
# # def predict(transaction: Transaction):
# #     if model_data is None:
# #         raise HTTPException(status_code=500, detail="ML Model not loaded.")

# #     try:
# #         # 1. Encode transaction type
# #         type_val = model_data["type_map"].get(transaction.type.upper())
# #         if type_val is None:
# #             raise ValueError(f"Invalid transaction type: {transaction.type}")

# #         # 2. Prepare features
# #         features = np.array([[transaction.step, type_val, transaction.amount]])

# #         # 3. Predict
# #         prediction = model_data["model"].predict(features)[0]
# #         probability = model_data["model"].predict_proba(features)[0]

# #         result = "Fraud" if prediction == 1 else "Not Fraud"
# #         confidence = float(max(probability))

# #         # 4. Gemini Explanation
# #         explanation = "Explanation unavailable."
# #         if client:
# #             try:
# #                 prompt = (
# #                     f"A fraud detection system predicted this transaction as '{result}' "
# #                     f"with confidence {confidence:.2f}. "
# #                     f"Details: Type={transaction.type}, Amount=${transaction.amount}, Time Step={transaction.step}. "
# #                     f"Explain in one simple, human-readable sentence why this might be fraud or safe."
# #                 )
# #                 # prompt = "Who is the ceo of google"
# #                 response = client.models.generate_content(
# #                     model="gemini-3-flash-preview", contents=prompt
# #                 )

# #                 if response.text:
# #                     explanation = response.text.strip()
# #                 else:
# #                     explanation = "AI could not generate an explanation."

# #             except Exception as e:
# #                 print(f"Gemini API error: {e}")
# #                 explanation = "AI explanation service is currently unavailable."

# #         return {
# #             "prediction": result,
# #             "confidence": round(confidence, 4),
# #             "explanation": explanation,
# #             "status": "success",
# #         }

# #     except Exception as e:
# #         raise HTTPException(status_code=400, detail=str(e))


# # if __name__ == "__main__":
# #     import uvicorn

# #     uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import os
# from fastapi.middleware.cors import CORSMiddleware
# from google import genai

# app = FastAPI(title="Financial Fraud Detection API")

# # ---------------- GEMINI SETUP ---------------- #

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# print("KEY DEBUG:", GEMINI_API_KEY)
# print("KEY LENGTH:", len(GEMINI_API_KEY) if GEMINI_API_KEY else None)
# client = None

# if GEMINI_API_KEY and GEMINI_API_KEY.strip():
#     try:
#         GEMINI_API_KEY = GEMINI_API_KEY.strip().strip('"').strip("'")
#         client = genai.Client(api_key=GEMINI_API_KEY)
#         print(f"Gemini initialized: {GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]} ✅")
#     except Exception as e:
#         print(f"Gemini initialization error: {e}")
#         client = None
# else:
#     print("⚠️ GEMINI_API_KEY not found. Running without AI explanations.")

# # ---------------- CORS ---------------- #
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ---------------- INPUT MODEL ---------------- #
# class Transaction(BaseModel):
#     step: int
#     type: str
#     amount: float


# # ---------------- LOAD ML MODEL ---------------- #
# model_data = None


# @app.on_event("startup")
# def load_model():
#     global model_data
#     model_path = "model/fraud_model.pkl"
#     if os.path.exists(model_path):
#         model_data = joblib.load(model_path)
#         print("ML Model loaded successfully.")
#     else:
#         print("❌ ML Model not found. Run train.py first.")


# # ---------------- ROUTES ---------------- #
# @app.get("/")
# def home():
#     return {"message": "Fraud Detection API is running. Use /predict"}


# @app.post("/predict")
# def predict(transaction: Transaction):
#     if model_data is None:
#         raise HTTPException(status_code=500, detail="ML Model not loaded.")

#     try:
#         # Encode type
#         type_val = model_data["type_map"].get(transaction.type.upper())
#         if type_val is None:
#             raise ValueError(f"Invalid transaction type: {transaction.type}")

#         # Features
#         features = np.array([[transaction.step, type_val, transaction.amount]])

#         # Prediction
#         prediction = model_data["model"].predict(features)[0]
#         probability = model_data["model"].predict_proba(features)[0]

#         result = "Fraud" if prediction == 1 else "Not Fraud"
#         confidence = float(max(probability))

#         # Gemini explanation
#         explanation = "Explanation unavailable."
#         if client:
#             try:
#                 prompt = (
#                     f"This transaction was predicted as '{result}' with confidence {confidence:.2f}. "
#                     f"Type: {transaction.type}, Amount: {transaction.amount}, Step: {transaction.step}. "
#                     f"Explain briefly why this is fraud or safe in one simple sentence."
#                 )

#                 response = client.models.generate_content(
#                     model="gemini-3-flash-preview",
#                     contents=prompt,
#                 )

#                 explanation = (
#                     response.text.strip()
#                     if response.text
#                     else "No explanation generated."
#                 )

#             except Exception as e:
#                 print(f"Gemini API error: {e}")
#                 explanation = "AI explanation service unavailable."

#         return {
#             "prediction": result,
#             "confidence": round(confidence, 4),
#             "explanation": explanation,
#             "status": "success",
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
from google import genai

app = FastAPI(title="Financial Fraud Detection API")

# ---------------- ENV ---------------- #
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    GEMINI_API_KEY = GEMINI_API_KEY.strip().strip('"').strip("'")
    print(f"Gemini key loaded: {GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]}")
else:
    print("⚠️ GEMINI_API_KEY not found")

# ---------------- CORS ---------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- INPUT MODEL ---------------- #
class Transaction(BaseModel):
    step: int
    type: str
    amount: float


# ---------------- LOAD MODEL ---------------- #
model_data = None


@app.on_event("startup")
def load_model():
    global model_data
    model_path = "model/fraud_model.pkl"

    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        print("ML Model loaded successfully.")
    else:
        print("❌ Model not found. Run train.py first.")


# ---------------- ROUTES ---------------- #
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running."}


@app.post("/predict")
def predict(transaction: Transaction):
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # 1. Encode type
        type_val = model_data["type_map"].get(transaction.type.upper())
        if type_val is None:
            raise ValueError(f"Invalid transaction type: {transaction.type}")

        # 2. Prepare input
        features = np.array([[transaction.step, type_val, transaction.amount]])

        # 3. Predict
        prediction = model_data["model"].predict(features)[0]
        probability = model_data["model"].predict_proba(features)[0]

        result = "Fraud" if prediction == 1 else "Not Fraud"
        confidence = float(max(probability))

        # 4. Gemini Explanation
        explanation = "Explanation unavailable."

        if GEMINI_API_KEY:
            try:
                # 🔥 Create client INSIDE request (stable)
                client = genai.Client(api_key=GEMINI_API_KEY)

                prompt = (
                    f"This transaction was predicted as '{result}' "
                    f"with confidence {confidence:.2f}. "
                    f"Type: {transaction.type}, Amount: {transaction.amount}, Step: {transaction.step}. "
                    f"Explain in one simple sentence why this is fraud or safe."
                )

                response = client.models.generate_content(
                    model="gemini-3-flash-preview",  # ❗ unchanged
                    contents=prompt,
                )

                if response.text:
                    explanation = response.text.strip()
                else:
                    explanation = "No explanation generated."

            except Exception as e:
                print("Gemini error:", e)
                explanation = "AI explanation failed."

        return {
            "prediction": result,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# === vision_server.py ===
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import httpx
import uvicorn
import logging
import os
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VISION_API_URL_1 = "https://api-vision-dev.sparc.bot/handup/predict"
VISION_API_URL_2 = "https://api-vision-dev.sparc.bot/face-recognition/predict"

def compute_intersection_area(box1, box2):
    x_left = max(box1['x1'], box2['x1'])
    y_top = max(box1['y1'], box2['y1'])
    x_right = min(box1['x2'], box2['x2'])
    y_bottom = min(box1['y2'], box2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0  # No overlap

    return (x_right - x_left) * (y_bottom - y_top)

def get_hand_raisers(handup_result, face_recognition_result, threshold=0.2):
    results = []

    for hand in handup_result:
        if hand['label'] == "hand-raising":
            max_overlap = 0
            matched_face = None
            match_set = []  # Initialize match_set here

            for face in face_recognition_result:
                intersection_area = compute_intersection_area(hand, face)
                face_area = (face['x2'] - face['x1']) * (face['y2'] - face['y1'])

                if face_area == 0:
                    continue

                overlap_ratio = intersection_area / face_area
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                    matched_face = face

            if matched_face:
                match_set.append({'name': matched_face['name'], 'hand_box': hand, 'face_box': matched_face, 'max_overlap': max_overlap})

            # Find the entry with the maximum overlap and add it to results
            if match_set:
                max_overlap_entry = max(match_set, key=lambda x: x['max_overlap'])
                if max_overlap_entry['max_overlap'] > threshold:
                    results.append(max_overlap_entry)

    return results

@app.post("/upload/")
async def receive_image(file: UploadFile = File(...), robot_id: str = Form(...)):
    try:
        file_bytes = await file.read()
        # Encode the image bytes to a Base64 string
        image_base64 = base64.b64encode(file_bytes).decode('utf-8')  # Convert bytes to Base64 string
        form_data = {'file': (file.filename, file_bytes, file.content_type)}

        # # Save the image to local disk
        # with open(f"./images/{file.filename}", "wb") as f:
        #     f.write(file_bytes)

        async with httpx.AsyncClient() as client:
            response1 = await client.post(VISION_API_URL_1, files=form_data)
            response2 = await client.post(VISION_API_URL_2, files=form_data)
            logger.info(f"\n----------------\n!!! handup_result: {response1.json().get('bounding_boxes', [])}\n----------------")
            logger.info(f"\n----------------\n!!! face_recognition_result: {response2.json().get('bounding_boxes', [])}\n----------------")

            # Ensure the results are lists
            handup_result_box = response1.json().get('bounding_boxes', [])  
            face_recognition_result_box = response2.json().get('bounding_boxes', [])
            face_recognition_result_att = response2.json().get('attendance', [])
            detect_user = get_hand_raisers(handup_result_box, face_recognition_result_box)
            logger.info(f"\n----------------\n!!! detect_user: {detect_user}\n----------------")

            # Prepare data to send to the new endpoint
            data_to_send = {
                "handup_result": handup_result_box if isinstance(handup_result_box, list) else [],
                "face_recognition_result": face_recognition_result_att if isinstance(face_recognition_result_att, list) else [],
                "robot_id": robot_id,
                "image_name": file.filename,
                "image": image_base64,
                "detect_user": detect_user if isinstance(detect_user, list) else [],
            }
            # logger.info(f"\n----------------\n✅ Data to send: {data_to_send}\n----------------")

            # Send data to the new endpoint with error handling
            try:
                await client.post("https://app-ragbackend-dev-wus-001.azurewebsites.net/vision/getData/", json=data_to_send)
                # await client.post("http://localhost:8000/vision/getData/", json=data_to_send)
                logger.info("\n----------------\n✅ Send data to RAG_backend successfully!\n----------------")
            except httpx.HTTPStatusError as e:
                logger.error(f"\n----------------\n❌ Failed to send data: {str(e)}\n----------------")
                return JSONResponse(content={"error": f"Failed to send data: {str(e)}"}, status_code=500)
            except Exception as e:
                logger.error(f"\n----------------\n❌ An error occurred while sending data: {str(e)}\n----------------")
                return JSONResponse(content={"error": f"An error occurred while sending data: {str(e)}"}, status_code=500)

            # Prepare the response to return
            result = {
                "handup_result": response1.json(),
                "face_recognition_result": response2.json()
            }

            return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # For local development only
    port = int(os.getenv("PORT", 8000))  # Default to 8000 for Azure compatibility
    uvicorn.run("vision_server:app", host="0.0.0.0", port=port, reload=True)
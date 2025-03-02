import cv2
import numpy as np

def preprocess_image(frame):
    # Convert to grayscale, blur, and detect edges.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged

def find_card_contour(edged):
    # Find contours and select a contour with 4 vertices having the maximum area.
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_contour = None
    max_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                card_contour = approx
    return card_contour

def classify_card(card_roi):
    # A simple placeholder classification based on mean intensity.
    # Replace this method with your actual detection logic or ML model.
    mean_val = np.mean(card_roi)
    if mean_val > 100:  # Adjust threshold as needed.
        return "Detected Card"
    else:
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        edged = preprocess_image(frame)
        card_contour = find_card_contour(edged)

        if card_contour is not None:
            cv2.drawContours(frame, [card_contour], -1, (0,255,0), 2)
            # Create mask and extract ROI based on the contour.
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [card_contour], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(card_contour)
            card_roi = cv2.bitwise_and(frame, frame, mask=mask)[y:y+h, x:x+w]
            card_gray = cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY)
            label = classify_card(card_gray)
        else:
            label = "No card detected"

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Card Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key to exit.
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

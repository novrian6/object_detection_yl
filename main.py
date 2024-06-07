import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import mediapipe as mp

def app():
    print(st.__version__)
    print(cv2.__version__)
    print(mp.__version__)

    st.header('Object Detection Web App')
    st.subheader('by Nova Novriansyah')
    st.write('Welcome!')

    # Load YOLO model
    model = YOLO('yolov8n.pt')
    object_names = list(model.names.values())

    # File uploader and form inputs
    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload video", type=['mp4'])
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=['person']) 
        min_confidence = st.slider('Confidence score', 0.0, 1.0, 0.5)
        submit_button = st.form_submit_button(label='Submit')
    
    if uploaded_file is not None and submit_button:
        # Create 'video' folder if it doesn't exist
        if not os.path.exists('video'):
            os.makedirs('video')
        
        # Save uploaded file to 'video' folder
        video_filepath = os.path.join('video', uploaded_file.name)
        with open(video_filepath, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Open video file
        video_stream = cv2.VideoCapture(video_filepath)
        if not video_stream.isOpened():
            st.error("Error: Could not open video file.")
            return
        
        # Get video properties
        width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_stream.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for better compatibility

        # Output video file
        output_filepath = video_filepath.replace('.mp4', '_output.mp4')
        out_video = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

        # Process video
        with st.spinner('Processing video...'):
            frame_count = 0
            while True:
                ret, frame = video_stream.read()
                if not ret:
                    break

                frame_count += 1  # Increment frame count

                # Perform object detection
                result = model(frame)
                for detection in result[0].boxes.data:
                    x0, y0 = int(detection[0]), int(detection[1])
                    x1, y1 = int(detection[2]), int(detection[3])
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name = model.names[cls]
                    label = f'{object_name} {score}'

                    if object_name in selected_objects and score > min_confidence:
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x0, y0 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Convert frame to RGB before writing
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out_video.write(frame)

            st.write(f"Processed {frame_count} frames.")  # Debug statement to verify frame count

        # Release resources
        video_stream.release()
        out_video.release()

        # Ensure the file is properly closed
        if not os.path.exists(output_filepath):
            st.error("Error: Output video file not found.")
            return

        st.success("Video processing complete!")
        st.write(f"Output video path: {output_filepath}")  # Debug statement to verify output path

        # Display the processed video
        st.video(output_filepath)

if __name__ == "__main__":
    app()

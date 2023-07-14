#include <opencv2/opencv.hpp>

int main()
{
    // Load the Haar cascade XML file for the object you want to detect
    std::string cascadePath = "haarcascade_upperbody.xml";
    cv::CascadeClassifier cascade;
    cascade.load(cascadePath);

    // Start the video capture
    cv::VideoCapture videoCapture(0);

    while (true)
    {
        // Capture frame-by-frame
        cv::Mat frame;
        videoCapture.read(frame);

        // Resize frame to half the original size
        cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

        // Convert the frame to grayscale for cascade detection
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Perform object detection
        std::vector<cv::Rect> objects;
        cascade.detectMultiScale(gray, objects, 1.05, 3, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        // Detect and label black or dark colored shirts
        for (const auto& rect : objects)
        {
            // Extract the region of interest (shirt area)
            cv::Mat shirtROI = frame(rect);

            // Convert the shirt region to HSV color space
            cv::Mat hsv;
            cv::cvtColor(shirtROI, hsv, cv::COLOR_BGR2HSV);

            // Define the range of black/dark colors in HSV
            cv::Scalar lowerBlack = cv::Scalar(0, 0, 0);
            cv::Scalar upperBlack = cv::Scalar(180, 255, 40);

            // Apply color thresholding to isolate black or dark regions
            cv::Mat blackMask;
            cv::inRange(hsv, lowerBlack, upperBlack, blackMask);

            // Count the number of black or dark pixels
            int blackPixelCount = cv::countNonZero(blackMask);

            // If a significant number of black or dark pixels are detected, label it as "Black/Dark Shirt"
            if (blackPixelCount > 300)
            {
                cv::rectangle(frame, rect, cv::Scalar(0, 0, 0), 2);
                cv::putText(frame, " ", cv::Point(rect.x, rect.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2);
                cv::putText(frame, "Black/Dark Shirt", cv::Point(rect.x, rect.y + rect.height + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2);
            }
        }

        // Display the resulting frame
        cv::imshow("Object Detection", frame);

        // Add a delay of 10 milliseconds
        cv::waitKey(10);

        // Exit the loop if 'q' is pressed
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    } 

    // Release the video capture and close the windows
    videoCapture.release();
    cv::destroyAllWindows();

    return 0;
}

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat outputWindow;
cv::VideoCapture inputVideo;
cv::VideoWriter outputVideo;
cv::Mat color[2];
cv::Mat gray[2];
cv::Mat win[9], legend[9];


int prev = 0, cur = 1;
int fps;
int miRecordCount = -1;
int minPixels;
int found_count = 0;
bool flag = false;

struct MIRecord {
    int prev, cur;
    cv::Mat gray;   // gray frame
    cv::Mat MEI1Fr;     // MEI between pre and cur frame
    cv::Mat MEI1Sec;   // MEI in past 1 second
};

void resizeImage(cv::Mat& oriVideo, int maxReso = 416) {
    int oriReso = std::max(oriVideo.rows, oriVideo.cols);
    if (oriReso <= maxReso) return;

    if (oriVideo.cols > oriVideo.rows) {
        cv::resize(oriVideo, oriVideo, cv::Size(maxReso, (int)((double)oriVideo.rows / oriVideo.cols * maxReso)));
    }
    else {
        cv::resize(oriVideo, oriVideo, cv::Size((int)((double)oriVideo.cols / oriVideo.rows * maxReso), maxReso));
    }
}

void createOutputWindow(cv::Mat inputVideo, cv::Mat win[], cv::Mat legend[], cv::Mat& outputWin, int margin = 15, int winPerRow = 3, int winPerCol = 3) {
    int rows = inputVideo.rows;
    int cols = inputVideo.cols;
    int winCount = 0, legendCount = 0;
    cv::Size outWinSize((cols + margin) * winPerRow - margin, (rows + margin) * winPerCol);

    outputWin = cv::Mat::ones(outWinSize, inputVideo.type()) * 64;

    for (int i = 0; i < winPerCol; i++) {
        for (int j = 0; j < winPerRow; j++) {
            win[winCount++] = outputWin(cv::Range((rows + margin) * i, (rows + margin) * i + rows),
                cv::Range((cols + margin) * j, (cols + margin) * j + cols));
        }
    }

    for (int bg = 20, i = 0; i < winPerCol; i++) // create the legend windows
        for (int j = 0; j < winPerRow; j++) {
            legend[legendCount] = outputWin(cv::Range((rows + margin) * i + rows, (rows + margin) * (i + 1)),
                cv::Range((cols + margin) * j, (cols + margin) * j + cols));
            legend[legendCount] = cv::Scalar(bg, bg, bg); // paint each in different colors
            bg += 30; // such that we can visually see the division from one to other
            if (bg > 80) bg = 20;
            legendCount++;
        }
}

void initiate() {
    // read in the first frame, resize and change it to grayscale
    inputVideo >> color[0];
    resizeImage(color[0]);
    color[0].copyTo(win[0]);
    createOutputWindow(color[0], win, legend, outputWindow);
    minPixels = color[0].cols * color[0].rows * 0.005;
    cv::cvtColor(color[0], gray[0], cv::COLOR_BGR2GRAY);
}

bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
    double area1 = contourArea(contour1);
    double area2 = contourArea(contour2);
    return area1 > area2; // Sort in descending order (largest to smallest)
}


int main() {
    std::string videoName = "C:/Users/Hp/Desktop/UCF_Crimes/Videos/Arson/Arson049_x264.mp4";
    std::string outputVideoName = "C:/Users/Hp/Desktop/UCF_Crimes/Videos/out/Test049_x264.mp4";
    std::string message = "";
    char str[256];
    inputVideo.open(videoName);
	if (!inputVideo.isOpened()) {
		std::cout << "Error opening video file" << std::endl;
		return 0;
	}
    
    std::ofstream outputFile;
    outputFile.open("C:/Users/Hp/Desktop/UCF_Crimes/Videos/out/output.txt", std::ios_base::app);
    if (!outputFile.is_open()) {
        std::cout << "Error opening text file" << std::endl;
        return 0;
    }
    
    fps = inputVideo.get(cv::CAP_PROP_FPS);

    initiate();

    outputVideo.open(outputVideoName, cv::VideoWriter::fourcc('D', 'I', 'V', '3'), inputVideo.get(cv::CAP_PROP_FPS),
        cv::Size(outputWindow.cols, outputWindow.rows), true);
    if (!outputVideo.isOpened()) {
        std::cout << "Could not open the output video for writing";
        system("pause");
        return 0;
    }

    cv::Mat diff1Fr, MEI1Fr, bright, bright2, bright3;
    MIRecord miRecord[40]; 
    miRecord[39].prev = -100;
    miRecord[39].cur = -100;

    cv::Mat MHI, MEI1Sec, temp, preMEI, hsv;
    MEI1Sec = cv::Mat::zeros(color[0].size(), CV_8U);

    while (true) {
        // Read a frame from the video
        inputVideo >> color[1];
        if (color[1].empty()) {
            break; // End of video
        }
        resizeImage(color[1]);

        color[1].copyTo(win[0]);
        sprintf_s(str, "Current frame (color) %d", cur);
        legend[0] = cv::Scalar(64);
        putText(legend[0], str, cv::Point(5, 11), 1, 1, cv::Scalar(250, 250, 250), 1);

        // convert to grayscale
        cv::cvtColor(color[1], gray[1], cv::COLOR_BGR2GRAY);

        // find MEI of prev and curr frame
        cv::absdiff(gray[0], gray[1], diff1Fr);
        MEI1Fr = diff1Fr > 20;
   

        if (cv::countNonZero(MEI1Fr) > minPixels){
            // compute MEI1Sec, MHI1Sec
            // circular array
            miRecordCount++;
            miRecordCount %= 40;

            miRecord[miRecordCount].prev = prev;
            miRecord[miRecordCount].cur = cur;
            MEI1Fr.copyTo(miRecord[miRecordCount].MEI1Fr);
            gray[1].copyTo(miRecord[miRecordCount].gray);

            MEI1Fr.copyTo(MEI1Sec);
            MEI1Fr.copyTo(preMEI);
            int valueSub = 0;

            for (int i = miRecordCount; ; ) {
                i--;
                if (i < 0) i = 39;
                if (cur - miRecord[i].prev > fps) break;

                MEI1Sec |= miRecord[i].MEI1Fr;
                temp = MEI1Sec - preMEI;
                MEI1Sec.copyTo(preMEI);
            }
            MEI1Sec.copyTo(miRecord[miRecordCount].MEI1Sec);

            cv::cvtColor(MEI1Sec, win[2], cv::COLOR_GRAY2BGR);
            sprintf_s(str, "MEI 1 second ");
            legend[2] = cv::Scalar(64);
            putText(legend[2], str, cv::Point(5, 11), 1, 1, cv::Scalar(250, 250, 250), 1);

            // find the bright area >128
            //bright = gray[1] > 200;
            bright = gray[1] > 128;
            cv::cvtColor(bright, win[1], cv::COLOR_GRAY2BGR);
            sprintf_s(str, "Bright area ");
            legend[1] = cv::Scalar(64);
            putText(legend[1], str, cv::Point(5, 11), 1, 1, cv::Scalar(250, 250, 250), 1);


            // intersect bright and MEI1sec
            cv::Mat blackhole;
            temp = bright & MEI1Sec;
            blackhole = temp & gray[1];
            cv::cvtColor(blackhole, win[3], cv::COLOR_GRAY2BGR);
            sprintf_s(str, "Intersect bright1 & MEI1Sec");
            legend[3] = cv::Scalar(64);
            putText(legend[3], str, cv::Point(5, 11), 1, 1, cv::Scalar(250, 250, 250), 1);

            // highlight the blackhole
            cv::Mat center;
            center = ~temp & bright;
            cv::cvtColor(center, win[4], cv::COLOR_GRAY2BGR);
            sprintf_s(str, "Possible center of flame");
            legend[4] = cv::Scalar(64);
            putText(legend[4], str, cv::Point(5, 11), 1, 1, cv::Scalar(250, 250, 250), 1);

            // Get the contours and sort it according to the area
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            findContours(center, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::sort(contours.begin(), contours.end(), compareContourAreas);

      
            // loop through the contours (10 largest), check for hurricane pattern
            std::vector<std::vector<cv::Point>> filtered_contours;
            cv::Mat dilated_mask, differenceMask, flame_pattern;
            cv::Mat contour_mask;
            int flame_pixels, mask_pixels;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            for (int i = 0; (i < contours.size()) && (i < 10); i++) {
                contour_mask = cv::Mat::zeros(center.size(), CV_8UC1);
                drawContours(contour_mask, contours, i, cv::Scalar(255), cv::FILLED);             
                dilate(contour_mask, dilated_mask, kernel, cv::Point(-1, -1), 3);
                absdiff(dilated_mask, contour_mask, differenceMask);
                flame_pattern = differenceMask & blackhole;
                flame_pixels = cv::countNonZero(flame_pattern);
                mask_pixels = cv::countNonZero(differenceMask);
                if (flame_pixels >= 0.5 * mask_pixels) {
                    filtered_contours.push_back(contours[i]);
                }
            }

            cv::Mat result;
            cv::Mat color_copy;
            color[1].copyTo(color_copy);
            // check if any filtered regions
            if (filtered_contours.size() > 0) {
                found_count++;

                result = cv::Mat::zeros(center.size(), CV_8UC1);
                drawContours(result, filtered_contours, -1, cv::Scalar(255), cv::FILLED);
                cv::cvtColor(result, win[5], cv::COLOR_GRAY2BGR);
                sprintf_s(str, "Filtered region");
                legend[5] = cv::Scalar(64);
                putText(legend[5], str, cv::Point(5, 11), 1, 1, cv::Scalar(250, 250, 250), 1);
                
                drawContours(color_copy, filtered_contours, -1, cv::Scalar(0, 0, 255), cv::FILLED);  
            }   
            color_copy.copyTo(win[6]);
            
       }
       if (found_count == 5 && !flag) {
           flag = true;
           message += videoName;
           message += "-";
           message += "Fire detected at frame: ";
           message += std::to_string(cur);
           message += "\n";
           sprintf_s(str, "Fire detected");
           legend[6] = cv::Scalar(64);
           putText(legend[6], str, cv::Point(5, 11), 1, 1, cv::Scalar(250, 250, 250), 1);
       }
       // reset the count every second
       if (cur % fps == 0) {
           found_count = 0;
       }
        // cur frame becomes prev frame 
        color[1].copyTo(color[0]);
        gray[1].copyTo(gray[0]);
        cur++;
        prev++;

        // Check for key press to exit
        if (cv::waitKey(30) == 27) { // Press ESC to exit
            break;
        }
        cv::imshow(videoName, outputWindow);
        outputVideo << outputWindow;
    }
    if (found_count < 5 && !flag) {
        message += videoName;
        message += "-";
        message += "No fire detected";
        message += "\n";
    }
    // Release the VideoCapture and close the window
    inputVideo.release();
    outputVideo.release();
    cv::destroyAllWindows();

    outputFile << message;
    outputFile.close();
	return 0;
}


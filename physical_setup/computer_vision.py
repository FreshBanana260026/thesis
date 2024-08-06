import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import csv
import torch
import joblib
import package_classifier
import copy

box_sizes = [([0.16, 0.12, 0.04], [0.17, 0.12, 0.04]),
             ([0.16, 0.08, 0.04], [0.17, 0.08, 0.04]),
             ([0.17, 0.09, 0.03], [0.17, 0.09, 0.03]),
             ([0.16, 0.14, 0.04], [0.16, 0.14, 0.04]),
             ([0.16, 0.06, 0.04], [0.17, 0.06, 0.04]),
             ([0.17, 0.05, 0.03], [0.17, 0.05, 0.03])]



class D435:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
        self.cfg = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.consistency_cnt = 0
        self.area = None
        self.area_buffer = []
        self.centroid_buffer_y = []
        self.centroid_buffer_x = []
        self.angle_buffer = []
        self.centroid = None
        self.angle = None
        self.real_centroid_meters = None
        self.model = package_classifier.PackageClassifier()
        self.model.load_state_dict(torch.load("package_classifier.pth"))
        self.model.eval()
        self.scalar = joblib.load("scaler.pkl")

    def get_entrinsics(self):
        color_intrinsics = self.cfg.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        #print("Color Intrinsics: ", color_intrinsics)
        #print("Color Intrinsics: ", color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy)
        return color_intrinsics

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return depth_frame, color_frame

    def check_consistency(self, area, centroid, angle):
        if self.area is None or self.centroid is None:
            self.area = area
            self.centroid = centroid
            return False
        #print("Area: ", self.area, "Centroid: ", self.centroid)
        #print("Area: ", area, "Centroid: ", centroid)
        if abs(self.area - area) < 100 and abs(self.centroid[0] - centroid[0]) < 10 and abs(self.centroid[1] - centroid[1]) < 10 and abs(self.angle - self.angle) < 5:
            # do a running average of the area
            self.area_buffer.append(area)
            self.centroid_buffer_x.append(centroid[0])
            self.centroid_buffer_y.append(centroid[1])
            self.angle_buffer.append(angle)
            
            self.consistency_cnt += 1
            self.area = sum(self.area_buffer) / len(self.area_buffer)
            self.centroid = np.array([sum(self.centroid_buffer_x) / len(self.centroid_buffer_x), sum(self.centroid_buffer_y) / len(self.centroid_buffer_y)])
            self.angle = sum(self.angle_buffer) / len(self.angle_buffer)
        else:
            self.consistency_cnt = 0

        if self.consistency_cnt > 50:
            self.consistency_cnt = 0
            return True

        return False

    def find_object(self):
        while True:
            depth_frame, color_frame = self.get_frame()

            depth_image = np.asanyarray(depth_frame.get_data())/1000
            color_image = np.asanyarray(color_frame.get_data())

            # Make image grayscale
            blue = color_image[:, :, 0]

            # Crop image to only picking area
            cropped_image = blue[145:295, 160:534]

            # Apply blurring and sharpening
            blur = cv.medianBlur(cropped_image, 5)
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv.filter2D(blur, -1, sharpen_kernel)

            # Threshold and morph
            thresh = cv.threshold(sharpen, 80, 255, cv.THRESH_BINARY_INV)[1]
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
            cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            cv.imshow('RealSense', thresh)
            cv.waitKey(1)
            # Find contours and filter using threshold area
            cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            min_area = 1500
            max_area = 7500
            rectangles = []
            area = 0
            for c in cnts:
                epsilon = 0.04 * cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, epsilon, True)
                area = cv.contourArea(c)
                if area > min_area and area < max_area and len(approx) == 4:
                    data_x, data_y, data_ratio = self.get_width_and_length(approx)
                    x,y,w,h = cv.boundingRect(c)
                    rectangles.append(approx)
                    break

            if len(rectangles) > 0:
                item_contour = rectangles[0]
            else:
                print("No item found")
                self.consistency_cnt = 0
                continue

            
            M = cv.moments(item_contour)
            angle_degrees = 0
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                angle = 0.5 * np.arctan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))
                angle_degrees = np.degrees(angle)
            else:
                cx, cy = 0, 0

            real_centroid_x = cx + 160
            real_centroid_y = cy + 145
            real_z = depth_image[real_centroid_y][real_centroid_x]#0.87

            angle = np.deg2rad(angle_degrees)

            center = np.array([real_centroid_x, real_centroid_y])
            arrow_size = 130
            point = np.array([real_centroid_x, real_centroid_y + arrow_size])
            point_y = np.array([real_centroid_x + arrow_size, real_centroid_y])

            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            rotated_point = center + np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), point - center)
            rotated_point = rotated_point.astype(int)

            rotated_point_y = center + np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), point_y - center)
            rotated_point_y = rotated_point_y.astype(int)

            output = cv.arrowedLine(color_image, (real_centroid_x, real_centroid_y), (rotated_point[0], rotated_point[1]), (255, 0, 0), 2, tipLength = 0.3)  #x-axis
            output = cv.arrowedLine(output, (real_centroid_x, real_centroid_y), (rotated_point_y[0], rotated_point_y[1]), (0, 0, 255), 2, tipLength = 0.3)  #y-axis

            if self.check_consistency(area, center, angle_degrees):
                print("Item found")
                color_intrinsics = self.get_entrinsics()
                self.real_centroid_meters = (((self.centroid[0] - color_intrinsics.ppx) * real_z) / color_intrinsics.fx, ((self.centroid[1] - color_intrinsics.ppy) * real_z) / color_intrinsics.fy, real_z)
                self.centroid_buffer_x.clear()
                self.centroid_buffer_y.clear()
                self.area_buffer.clear()
                self.angle_buffer.clear()
            else:
                self.area = area
                color_intrinsics = self.get_entrinsics()
                self.real_centroid_meters = (((center[0] - color_intrinsics.ppx) * real_z) / color_intrinsics.fx, ((center[1] - color_intrinsics.ppy) * real_z) / color_intrinsics.fy, real_z)
                self.centroid = center
                self.angle = angle_degrees
                continue

            scaled_datapoint = self.scalar.transform([[data_x, data_y, data_ratio, self.area]]).reshape(1, -1)
            print(type(torch.tensor(scaled_datapoint)))
            classified_size = self.model(torch.tensor(scaled_datapoint).float())
            box_size = box_sizes[classified_size.argmax()]
            print("Area: ", area, "Centroid: ", self.real_centroid_meters, "Angle: ", angle_degrees, "Box size: ", box_size)
            #input("Press Enter to continue...")

            # cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            # cv.imshow('RealSense', output)
            # cv.waitKey(1)

            return box_size, self.real_centroid_meters, self.angle


    def find_width_height_area(self):
        while True:
            depth_frame, color_frame = self.get_frame()

            depth_image = np.asanyarray(depth_frame.get_data())/1000
            color_image = np.asanyarray(color_frame.get_data())

            # Make image grayscale
            #gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
            blue = color_image[:, :, 0]

            # Crop image to only picking area
            cropped_image = blue[145:295, 160:534]

            # Apply blurring and sharpening
            blur = cv.medianBlur(cropped_image, 5)
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv.filter2D(blur, -1, sharpen_kernel)

            # Threshold and morph
            thresh = cv.threshold(sharpen, 50, 255, cv.THRESH_BINARY_INV)[1]
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

            # Find contours and filter using threshold area
            cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            cv.imshow('RealSense', close)
            cv.waitKey(1)
            min_area = 1500
            max_area = 7500
            rectangles = []
            area = 0
            for c in cnts:
                epsilon = 0.04 * cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, epsilon, True)
                area = cv.contourArea(c)
                if area > min_area and area < max_area and len(approx) == 4:
                    data_x, data_y, data_ratio = self.get_width_and_length(approx)
                    print(data_x, data_y, data_ratio, area)
                    with open("/home/pmkz/Desktop/packages.csv", mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([data_x, data_y, data_ratio, area, 6])
                    x,y,w,h = cv.boundingRect(c)
                    #print(approx)
                    rectangles.append(approx)

                    break
            if len(rectangles) > 0:
                item_contour = rectangles[0]
            else:
                #print("No item found")
                continue

            M = cv.moments(item_contour)
            angle_degrees = 0
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                angle = 0.5 * np.arctan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))
                angle_degrees = np.degrees(angle)
            else:
                cx, cy = 0, 0

            real_centroid_x = cx + 160
            real_centroid_y = cy + 145
            real_z = depth_image[real_centroid_y][real_centroid_x]#0.87

            angle = np.deg2rad(angle_degrees)

            center = np.array([real_centroid_x, real_centroid_y])
            arrow_size = 130
            point = np.array([real_centroid_x, real_centroid_y + arrow_size])
            point_y = np.array([real_centroid_x + arrow_size, real_centroid_y])

            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            rotated_point = center + np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), point - center)
            rotated_point = rotated_point.astype(int)

            rotated_point_y = center + np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), point_y - center)
            rotated_point_y = rotated_point_y.astype(int)

            output = cv.arrowedLine(color_image, (real_centroid_x, real_centroid_y), (rotated_point[0], rotated_point[1]), (255, 0, 0), 2, tipLength = 0.3)  #x-axis
            output = cv.arrowedLine(output, (real_centroid_x, real_centroid_y), (rotated_point_y[0], rotated_point_y[1]), (0, 0, 255), 2, tipLength = 0.3)  #y-axis

            if self.check_consistency(area, center, angle_degrees):
                print("Item found")
                color_intrinsics = self.get_entrinsics()
                self.real_centroid_meters = (((self.centroid[0] - color_intrinsics.ppx) * real_z) / color_intrinsics.fx, ((self.centroid[1] - color_intrinsics.ppy) * real_z) / color_intrinsics.fy, real_z)
                self.centroid_buffer_x.clear()
                self.centroid_buffer_y.clear()
                self.area_buffer.clear()
                self.angle_buffer.clear()
            else:
                self.area = area
                color_intrinsics = self.get_entrinsics()
                self.real_centroid_meters = (((center[0] - color_intrinsics.ppx) * real_z) / color_intrinsics.fx, ((center[1] - color_intrinsics.ppy) * real_z) / color_intrinsics.fy, real_z)
                self.centroid = center
                self.angle = angle_degrees
                print("Item not found")
                continue

    def show_frames(self):
        i = 0
        while True:
            depth_frame, color_frame = self.get_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            #if i == 50:
            #    cv.imwrite("depth_image.png", depth_image)
            #    cv.imwrite("color_image.png", color_image)
            #    break
            # downsize images
            #depth_image = depth_image[140:280, 200:500]
            #color_image = color_image[140:280, 200:500]
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            print(depth_colormap_dim)
            print(color_colormap_dim)

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))
            i += 1
            # Show images
            cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            cv.imshow('RealSense', images)
            cv.waitKey(1)

    def make_box_dataset(self):
        for box in box_sizes:
            for i in range(10):
                print("Move box: ", box)
                input("Press Enter to continue...")
                self.find_width_height_area()

    def get_width_and_length(self, approx):
        # Extract the four points
        pts = approx.reshape(4, 2)

        # Calculate distances between each pair of points
        distances = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i+1) % 4]  # Ensures that the last point connects to the first
            distance = np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
            distances.append(distance)

        # Sort distances. The two smaller ones are widths, and the larger ones are lengths
        distances.sort()
        y = (distances[0] + distances[1]) / 2
        x = (distances[2] + distances[3]) / 2
        x = round(x, 3)
        y = round(y, 3)
        ratio = x / y
        return x, y, ratio
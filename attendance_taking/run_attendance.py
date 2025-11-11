import cv2
import os
import argparse
from config import Config
from detectors.yolo_detector import YOLODetector
from processors.grid_manager import GridManager
from processors.assignment import AssignmentProcessor
from models.classroom import Classroom
from utils.visualization import draw_detections, draw_attendance_status
from utils.file_io import save_attendance_report, save_visualization

class AttendanceSystem:
    def __init__(self, model_path=None):
        self.config = Config
        model_to_use = model_path or Config.MODEL_PATH
        self.detector = YOLODetector(model_to_use, Config.CONFIDENCE_THRESHOLD)
        self.grid_manager = GridManager(Config.CLASSROOM_ROWS, Config.CLASSROOM_COLS)
        self.assignment_processor = AssignmentProcessor(
            Config.MAX_DISTANCE_RATIO, Config.FALLBACK_DISTANCE
        )
        self.classroom = Classroom(Config.CLASSROOM_ROWS, Config.CLASSROOM_COLS)
    
    def process_image(self, image_path, save_output=True):
        """Process a single image and update attendance"""
        print(f"Processing: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            return None
        
        # Detect objects
        persons, tables = self.detector.detect(image_path)
        print(f"Found {len(persons)} persons and {len(tables)} tables")
        
        # Create table grid
        table_grid = self.grid_manager.create_table_grid(tables)
        self.classroom.table_grid = table_grid
        
        if table_grid is not None:
            # Assign persons to tables
            assignments = self.assignment_processor.assign_persons_to_tables(
                persons, table_grid, (Config.CLASSROOM_ROWS, Config.CLASSROOM_COLS)
            )
            
            # Update attendance
            self.classroom.update_attendance(assignments)
            print(f"Assigned {len(assignments)} persons to tables")
        else:
            print("Warning: No table grid created")
        
        # Generate outputs
        if save_output:
            self._generate_outputs(image_path, persons, tables)
        
        return self.classroom
    
    def process_folder(self, folder_path):
        """Process all images in a folder"""
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found")
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        results = []
        for image_file in image_files:
            result = self.process_image(image_file, save_output=True)
            if result:
                results.append(result)
        
        print(f"Processed {len(results)} images from folder")
        return results
    
    def _generate_outputs(self, image_path, persons, tables):
        """Generate output files and visualizations"""
        # Save reports
        json_path, csv_path = save_attendance_report(self.classroom)
        print(f"Saved reports: {json_path}, {csv_path}")
        
        # Create and save visualization
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = draw_detections(image, persons, tables)
                image = draw_attendance_status(image, self.classroom)
                viz_path = save_visualization(image, image_path)
                print(f"Saved visualization: {viz_path}")
            else:
                print("Warning: Could not load image for visualization")
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def print_attendance(self):
        """Print attendance matrix in readable format"""
        print("\n" + "="*50)
        print("ATTENDANCE REPORT")
        print("="*50)
        
        print("\nAttendance Matrix:")
        print("+" + "---+" * self.classroom.cols)
        
        for row in range(self.classroom.rows):
            row_str = "|"
            for col in range(self.classroom.cols):
                status = "P" if self.classroom.attendance[row, col] == 1 else "A"
                row_str += f" {status} |"
            print(row_str)
            print("+" + "---+" * self.classroom.cols)
        
        report = self.classroom.get_report()
        print(f"\nSummary:")
        print(f"Present: {report['present']}")
        print(f"Absent: {report['absent']}")
        print(f"Total Seats: {report['total_seats']}")
        print(f"Attendance: {report['percentage']:.1f}%")
        print("="*50)

def main():
    """Main function to run the attendance system"""
    parser = argparse.ArgumentParser(description='Classroom Attendance System')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--model', type=str, help='Path to custom model')
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        print("Please provide either --image or --folder argument")
        return
    
    # Initialize system
    system = AttendanceSystem(model_path=args.model)
    
    if args.image:
        # Process single image
        classroom = system.process_image(args.image)
        if classroom:
            system.print_attendance()
    
    elif args.folder:
        # Process folder
        results = system.process_folder(args.folder)
        if results:
            # Print summary of last processed image
            system.print_attendance()

if __name__ == "__main__":
    main()
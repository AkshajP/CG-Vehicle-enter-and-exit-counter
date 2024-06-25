### Tracker Class

#### Initialization (`__init__` method)

1. **`self.center_points = {}`**: Initializes an empty dictionary to store the center points of detected objects, where the keys are object IDs and the values are tuples representing the coordinates of the center points.
2. **`self.id_count = 0`**: Initializes an ID counter to assign unique IDs to newly detected objects.

#### Update Method (`update` method)

The `update` method is called to update the tracker with new object detections. It takes a list of bounding box coordinates (`objects_rect`) as input and outputs a list of bounding boxes with associated object IDs.

1. **`objects_bbs_ids = []`**: Initializes an empty list to store bounding boxes and their corresponding IDs.

2. **Get Center Point of New Object**: For each rectangle in `objects_rect`:

   - Extracts the coordinates (`x`, `y`) and dimensions (`w`, `h`) of the bounding box.
   - Calculates the center point (`cx`, `cy`) of the bounding box.

3. **Detect Same Object**: Checks if the current object has been detected previously:

   - Iterates through the stored center points in `self.center_points`.
   - Calculates the Euclidean distance between the new center point (`cx`, `cy`) and the stored center point.
   - If the distance is less than 35 pixels, it considers the object as the same and updates its center point in `self.center_points`.
   - Appends the bounding box and the corresponding ID to `objects_bbs_ids`.
   - Sets `same_object_detected` to `True` and breaks out of the loop.

4. **New Object Detection**: If the object is new (`same_object_detected` is `False`):

   - Assigns a new ID to the object using `self.id_count`.
   - Updates `self.center_points` with the new center point and ID.
   - Appends the bounding box and the new ID to `objects_bbs_ids`.
   - Increments `self.id_count`.

5. **Clean Up Center Points**:

   - Creates a new dictionary `new_center_points` to store the current valid center points.
   - Iterates through `objects_bbs_ids` and updates `new_center_points` with the current valid center points and their IDs.
   - Updates `self.center_points` with `new_center_points` to remove any IDs that are no longer used.

6. **Return**: Returns the list of bounding boxes with their associated IDs (`objects_bbs_ids`).

### Example Usage

This class can be used in a computer vision project where objects are detected frame by frame. The `update` method will help in maintaining the identity of objects across frames by assigning and updating unique IDs.

This code helps maintain continuity in object tracking by assigning consistent IDs to detected objects as they move through the frames of a video.

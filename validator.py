import numpy as np

class DepthValidator:

    def calculate_pixel_length(object_length, distance_to_camera, resolution_width, field_of_view):
        # Calculate Angular Resolution
        angular_resolution = resolution_width / field_of_view

        # Calculate Angular Size
        angular_size_radians = 2 * np.arctan(object_length / (2 * distance_to_camera))
        angular_size_degrees = np.degrees(angular_size_radians)

        # Convert Angular Size to Pixels
        pixel_length = angular_size_degrees * angular_resolution

        return pixel_length


    def calculate_distance_to_camera(object_length, pixel_length, resolution_width, field_of_view):
        # Calculate Angular Resolution
        angular_resolution = resolution_width / field_of_view

        # Calculate Angular Size
        angular_size_degrees = pixel_length / angular_resolution

        # Calculate Distance to Camera
        distance_to_camera = (object_length / 2) / np.tan(np.radians(angular_size_degrees / 2))

        return distance_to_camera

    def is_valid_depth(stereo_depth, object_length, pixel_length, resolution_width, field_of_view, min_object_size=5000, max_object_size=8000): #size in milimeters
        minopticaldepth = DepthValidator.calculate_distance_to_camera(min_object_size, pixel_length, resolution_width, field_of_view)
        maxopticaldepth = DepthValidator.calculate_distance_to_camera(max_object_size, pixel_length, resolution_width, field_of_view)
        print(f'max:{maxopticaldepth}')
        print(f'min:{minopticaldepth}')
        inRange = (min_object_size <= stereo_depth) & (stereo_depth <= max_object_size)

        return inRange


if __name__ == '__main__':
    # Example usage
    object_length = 1000  # in milimeters
    distance_to_camera = 5000  # in milimeters
    resolution_width = 3840  # pixels (4k resolution)
    field_of_view = 69  # degrees

    pixel_length = DepthValidator.calculate_pixel_length(object_length, distance_to_camera, resolution_width, field_of_view)
    print("Pixel length of a 1-meter object at 5 meters distance:", pixel_length)

    # Example usage
    object_length = 1000  # in milimeters
    pixel_length = 636.26  # pixels
    resolution_width = 3840  # pixels (4k resolution)
    field_of_view = 69  # degrees

    distance_to_camera = DepthValidator.calculate_distance_to_camera(object_length, pixel_length, resolution_width, field_of_view)
    print("Distance to camera for a 1-meter object with pixel length", pixel_length, ":", distance_to_camera, "milimeters")

    #example for validity check
    stereo_depth = 1100
    object_length = 1000
    pixel_length = 636.26
    resolution_width = 3840
    field_of_view = 69
    inRange = DepthValidator.is_valid_depth(stereo_depth, object_length, pixel_length, resolution_width, field_of_view, min_object_size=900, max_object_size=1100)
    print(f'stereo depth validity:{inRange}')

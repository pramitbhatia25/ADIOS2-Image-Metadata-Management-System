# Image Metadata Management System
This project is designed to manage image data and metadata for experiments. It offers functionalities to insert, query, extract, and delete data using a combination of ADIOS2 for handling large datasets, OpenCV for loading and processing images and SQLite for metadata management.

## Features

### Data Insertion:

- Converts raw image data into ADIOS BP format.
- Stores metadata and BP file paths in a SQLite database.
- Metadata can be manually entered, AI-generated based on image content, or custom provided.

### Data Query:

 - Retrieves and displays metadata for experiments based on the experiment name.

### Data Extraction:

- Converts BP format data back to raw images.
- Outputs metadata along with the extracted images.

### Data Deletion:

- Removes experiment data and corresponding BP files from the database.

## Getting Started
### Prerequisites
 - C++17 or higher
 - CMake for build management
 - ADIOS2 for data handling
 - SQLite3 for database management
 - OpenCV for image processing
 - YOLOv5 model for object detection

### Building the Project

Clone the repository:

```console
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

Create a build directory and navigate to it:

```console
mkdir build
cd build
```

Configure and build the project using CMake:

```console
cmake ..
make
```

### Contributing
Contributions are welcome! Please fork the repository and submit pull requests.


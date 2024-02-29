/*
# On execution, select from the options provided.
# To insert data, enter 1.
# To query data, enter 2.
# To extract data, enter 3.

# Data Insert:
# Enter metadata and a link to raw image data.
# 'Experiment Name' must be a unique field.
# The raw image data will be converted into adios bp file, the location of which will be stored in the database along with the metadata.

# Data Query:
# Query parameter is 'Experiment Name', thus enter the name of the experiment to query.

# Data Extract:
# The adios bp data will be converted into raw images, and the metadata will be shown along with output location.

*/

#include <iostream>
#include <sqlite3.h>
#include <fstream>
#include <vector>
#include <experimental/filesystem>
#include <adios2.h>
#include <opencv2/opencv.hpp>

namespace fs = std::experimental::filesystem;

struct ConversionResult {
    std::string outputPath;
    std::string metadataContent;
};

ConversionResult convert_images(const std::string& experimentName, const std::string& rawPath) {
	int rank, size;
	rank = 0;
	size = 1;
	
	// Initializes ADIOS Object
	
	adios2::ADIOS adios;
	adios2::IO bpIO = adios.DeclareIO("image_write");
	bpIO.SetEngine("bp3");

	// Checks if rawpath exists
	
	if (!fs::exists(rawPath) || !fs::is_directory(rawPath)) {
		std::cout << "Error: The specified path does not exist or is not a directory." << std::endl;
		return {"Error","Error"};
	}

	// Defines the output path and opens a .bp file at that location using ADIOS
	
	std::string outputPath = "/home/pbhatia4/Desktop/Adios2C-Implementation/ImageBPFiles/" + experimentName + "/images.bp";
	adios2::Engine bpFileWriter = bpIO.Open(outputPath, adios2::Mode::Write);

	// Collects all fileNames in rawpath
	std::vector<std::string> fileNames;
	for (const auto& entry : fs::directory_iterator(rawPath)) {
		if (fs::is_regular_file(entry.status())) {
		    fileNames.push_back(entry.path().filename());
		}
	}

	// Iterates through those fileNames, reads data and creates a variable for each image inside the .bp file
	bool found = false;

	for (const auto& fileName : fileNames) {
	        if (fileName == "metadata.txt") {
	        	found = true;
	        	continue;
	        }

		std::string imagePath = rawPath + fileName;
		cv::Mat image = cv::imread(imagePath);

		if (image.empty()) {
		    std::cerr << "Error: Couldn't open or read the image at " << imagePath << std::endl;
		return {"Error","Error"};
		}

		if (image.channels() == 1) {
		    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
		}

		const int height = image.rows;
		const int width = image.cols;
		const int channels = image.channels();

		auto ioImage = bpIO.DefineVariable<uint8_t>(fileName, {size * height, width, channels}, {rank * height, 0, 0}, {height, width, channels}, false);
	
		std::cout << "Writing " << fileName << std::endl;
		bpFileWriter.Put(ioImage, image.data, adios2::Mode::Sync);
	}

    	if(!found) {
		std::string metadataFilePath = rawPath + "metadata.txt";
		std::ifstream metadataFile(metadataFilePath);
		std::string metadataContent = "";

		bool validChoice = false;

		do {
			std::cout << "\nMetadata File Not Found!\nSelect an option below:\n";
			std::cout << "1) Use empty metadata file\n2) AI generate metadata based on images\n3) Add custom metadata file content\n";

			int userChoice;
			std::cin >> userChoice;
			std::ofstream metadataFile(metadataFilePath);  // Reopen the file stream
			switch (userChoice) {
			case 1:
				metadataContent = "";
				validChoice = true;
				
				if (metadataFile.is_open()) {
					metadataFile << metadataContent;
					std::cout << "Empty metadata File created and content written successfully!" << std::endl;
				} else {
					std::cerr << "Error creating/writing metadata file!" << std::endl;
				}
										
				break;
			case 2:
				// AI generate metadata based on images
				// Implement AI generation logic here if needed
				// metadataContent = generateMetadataAI();
				metadataContent = "";
				validChoice = true;
				
				if (metadataFile.is_open()) {
					metadataFile << metadataContent;
					std::cout << "Empty metadata File created and content written successfully!" << std::endl;
				} else {
					std::cerr << "Error creating/writing metadata file!" << std::endl;
				}
				break;
			case 3:
				std::cout << "Enter custom metadata content: ";
				std::cin.ignore();  // Clear input buffer
				std::getline(std::cin, metadataContent);
				validChoice = true;

				if (metadataFile.is_open()) {
					metadataFile << metadataContent;
					std::cout << "\nCustom metadata File created and content written successfully!" << std::endl;
				} else {
					std::cerr << "Error creating/writing metadata file!" << std::endl;
				}

				break;
			default:
				std::cerr << "Invalid choice. Please enter a valid choice." << std::endl;
			}
			metadataFile.close();  // Close the file stream after writing					
		} while (!validChoice);   	
    	}

	fileNames.clear();

	for (const auto& entry : fs::directory_iterator(rawPath)) {
		if (fs::is_regular_file(entry.status())) {
		    fileNames.push_back(entry.path().filename());
		}
	}

	std::string metadataContent = "";
	for (const auto& fileName : fileNames) {
		if (fileName == "metadata.txt") {
			std::string metadataFilePath = rawPath + "metadata.txt";
			std::ifstream metadataFile(metadataFilePath);
			if (metadataFile.is_open()) {
				std::stringstream buffer;
				buffer << metadataFile.rdbuf();
				metadataContent = buffer.str();
				std::cout << "\nFound Metadata!\nMetadata Content: \n" << metadataContent << std::endl;
				adios2::Attribute<std::string> metadataAttribute = bpIO.DefineAttribute<std::string>("metadata", metadataContent);
				metadataFile.close();  // Close the file stream				
			}
	
		}
	}	

    
	bpFileWriter.Close();
	return {outputPath, metadataContent};
}

void insertDataToDatabase(const std::string& authorName, const std::string& experimentName, const std::string& adiosOutputPath, const std::string& metadataContent) {

	// Purpose is to store authorName, experimentName, adiosOutputPath inside the database
	
	sqlite3* db;
	int rc = sqlite3_open("data.db", &db);

	if (rc) {
		std::cerr << "Error: Can't open database: " << sqlite3_errmsg(db) << std::endl;
		return;
	}

	// Creates table if it doesnt already exist

	std::string createTableQuery = "CREATE TABLE IF NOT EXISTS experiment_data ("
		                   "id INTEGER PRIMARY KEY AUTOINCREMENT, "
		                   "author_name TEXT, "
		                   "experiment_name TEXT UNIQUE, "
		                   "adios_image_path TEXT,"
		                   "metadataContent TEXT);";
	
	// The function signature of the sqlite exec API expects 3 additional parameters that we dont need. These are callback function, arg to callback, and an error message

	rc = sqlite3_exec(db, createTableQuery.c_str(), nullptr, nullptr, nullptr);

	if (rc != SQLITE_OK) {
		std::cerr << "Error: Failed to create table: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
		return;
	}

	std::string sqlScript = "INSERT INTO experiment_data (author_name, experiment_name, adios_image_path, metadataContent) VALUES (?, ?, ?, ?);";
	sqlite3_stmt* stmt;
	rc = sqlite3_prepare_v2(db, sqlScript.c_str(), -1, &stmt, nullptr);

	if (rc != SQLITE_OK) {
		std::cerr << "Error: Failed to prepare query: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
		return;
	}

	rc = sqlite3_bind_text(stmt, 1, authorName.c_str(), -1, SQLITE_STATIC);

	if (rc != SQLITE_OK) {
		std::cerr << "Error: Failed to bind parameters: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_finalize(stmt);
		sqlite3_close(db);
		return;
	}

	rc = sqlite3_bind_text(stmt, 2, experimentName.c_str(), -1, SQLITE_STATIC);

	if (rc != SQLITE_OK) {
		std::cerr << "Error: Failed to bind parameters: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_finalize(stmt);
		sqlite3_close(db);
		return;
	}

	rc = sqlite3_bind_text(stmt, 3, adiosOutputPath.c_str(), -1, SQLITE_STATIC);

	if (rc != SQLITE_OK) {
		std::cerr << "Error: Failed to bind parameters: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_finalize(stmt);
		sqlite3_close(db);
		return;
	}

	rc = sqlite3_bind_text(stmt, 4, metadataContent.c_str(), -1, SQLITE_STATIC);

	if (rc != SQLITE_OK) {
		std::cerr << "Error: Failed to bind parameters: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_finalize(stmt);
		sqlite3_close(db);
		return;
	}

	rc = sqlite3_step(stmt);

	if (rc != SQLITE_DONE) {
		std::cerr << "Error: Failed to execute query: " << sqlite3_errmsg(db) << std::endl;
	}

	sqlite3_finalize(stmt);
	sqlite3_close(db);
}

bool checkdb(const std::string& experimentName) {
    sqlite3* db;
    int exit = 0;
    exit = sqlite3_open("data.db", &db);

    if (exit) {
        std::cout << "Error: Can't open database: \n" << sqlite3_errmsg(db) << std::endl;
        std::exit(0);
    }

	std::string createTableQuery = "CREATE TABLE IF NOT EXISTS experiment_data ("
		                   "id INTEGER PRIMARY KEY AUTOINCREMENT, "
		                   "author_name TEXT, "
		                   "experiment_name TEXT UNIQUE, "
		                   "adios_image_path TEXT,"
		                   "metadataContent TEXT);";
	
    int rc = sqlite3_exec(db, createTableQuery.c_str(), nullptr, nullptr, nullptr);

    if (rc != SQLITE_OK) {
        std::cout << "Error: Failed to create table: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        std::exit(0);        
    }

    std::string query = "SELECT experiment_name FROM experiment_data WHERE experiment_name = ?;";
    sqlite3_stmt* stmt;

    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cout << "Error: Failed to prepare query: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        std::exit(0);        
    }

    rc = sqlite3_bind_text(stmt, 1, experimentName.c_str(), -1, SQLITE_STATIC);

    if (rc != SQLITE_OK) {
        std::cout << "Error: Failed to bind parameters: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        std::exit(0);        
    }

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    return (rc == SQLITE_ROW);
}


void insertDataAndGetPath() {
	std::string experimentName;
	std::string rawImagesPath;
	std::string authorName;

	std::cout << "Enter Experiment Name: ";
	std::cin >> experimentName;

	if (checkdb(experimentName)) {
		std::cout << "Experiment already exists in the database!" << std::endl;
		return;
	}

	std::cout << "Enter Author Name: ";
	std::cin >> authorName;

	std::cout << "Enter path to the directory containing raw images: ";
	std::cin >> rawImagesPath;
	
	std::cout << "\n";
	std::string outputPath;
	std::string metadataContent;
	

	ConversionResult result = convert_images(experimentName, rawImagesPath);
	outputPath = result.outputPath;
	metadataContent = result.metadataContent;

	if(outputPath != "Error") {
		std::cout << "\nBP File Location: " << outputPath;
		insertDataToDatabase(authorName, experimentName, outputPath, metadataContent);
	}
	else {
		std::cout << "Error!";
		return;
	}
}

bool queryAllData() {
    sqlite3* db;
    int exit = 0;
    exit = sqlite3_open("data.db", &db);

    if (exit) {
        std::cerr << "Error: Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

	std::string createTableQuery = "CREATE TABLE IF NOT EXISTS experiment_data ("
		                   "id INTEGER PRIMARY KEY AUTOINCREMENT, "
		                   "author_name TEXT, "
		                   "experiment_name TEXT UNIQUE, "
		                   "adios_image_path TEXT,"
		                   "metadataContent TEXT);";

    int rc = sqlite3_exec(db, createTableQuery.c_str(), nullptr, nullptr, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Error: Failed to create table: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return false;
    }

    std::string query = "SELECT * FROM experiment_data;";
    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Error: Failed to prepare query: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return false;
    }

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        std::cout << "Author Name: " << sqlite3_column_text(stmt, 1) << std::endl;
        std::cout << "Experiment Name: " << sqlite3_column_text(stmt, 2) << std::endl;
        std::cout << "Adios Image Path: " << sqlite3_column_text(stmt, 3) << std::endl;
        std::cout << "MetaData: \n" << sqlite3_column_text(stmt, 4) << std::endl;
        std::cout << "-----------------------------" << std::endl;
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    return true;
}

void extractImages() {
    sqlite3* db;
    int exit = 0;
    exit = sqlite3_open("data.db", &db);
	
    if (exit) {
        std::cerr << "Error: Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    queryAllData();

    std::string experimentName;
    std::cout << "Enter Experiment Name to Extract Images: ";
    std::cin >> experimentName;
    
    std::string query = "SELECT adios_image_path FROM experiment_data WHERE experiment_name = ?;";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Error: Failed to prepare query: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return;
    }

    rc = sqlite3_bind_text(stmt, 1, experimentName.c_str(), -1, SQLITE_STATIC);

    if (rc != SQLITE_OK) {
        std::cerr << "Error: Failed to bind parameters: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return;
    }

    rc = sqlite3_step(stmt);

    if (rc != SQLITE_ROW) {
        std::cerr << "Error: Experiment not found in the database." << std::endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return;
    }

    std::string adiosImagePath(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    std::cout << "BP File Path: " << adiosImagePath << "\n\n";
	
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    adios2::ADIOS adios;
    adios2::IO bpIO = adios.DeclareIO("image_read");
    adios2::Engine bpReader = bpIO.Open(adiosImagePath, adios2::Mode::Read);


    const std::map<std::string, adios2::Params> varss = bpIO.AvailableVariables();
    std::string output_folder = "/home/pbhatia4/Desktop/Adios2C-Implementation/Data-Output/" + experimentName + "/";
    fs::create_directories(output_folder);

    for (const auto& variable_name : varss) {
        auto bpImage = bpIO.InquireVariable(variable_name.first);
        if (bpImage) {
            std::cout << "Reading " << variable_name.first << std::endl;
            auto shape = bpImage.Shape();
            size_t height = shape[0];
            size_t width = shape[1];
            size_t channels = shape[2];

            std::vector<uint8_t> myImage(height * width * channels);

            bpImage.SetSelection({{0, 0, 0}, {height, width, channels}});
            bpReader.Get(bpImage, myImage.data(), adios2::Mode::Sync);

            if (channels < 3) {
                myImage.resize(height * width * 3);
                std::fill(myImage.begin() + channels, myImage.end(), 0);
            }

            cv::Mat image(height, width, (channels == 3) ? CV_8UC3 : CV_8UC1, myImage.data());
            cv::imwrite(output_folder + variable_name.first, image);
        }
    }

	adios2::Attribute<std::string> metadataAttribute;

	// Inquire the attribute from the BP file
	metadataAttribute = bpIO.InquireAttribute<std::string>("metadata");

	// Check if the attribute exists
	if (metadataAttribute) {
	    // Retrieve the data associated with the attribute
	    const std::vector<std::string>& metadataValues = metadataAttribute.Data();

	    // Process the retrieved data as needed
	    if (!metadataValues.empty()) {
		std::string metadataValue = metadataValues[0];
		std::cout << "\nInquired Attribute Value: \n" << metadataValue << std::endl;
		std::ofstream metadataFile(output_folder + "metadata.txt");
		metadataFile << metadataValue;
		metadataFile.close();
		std::cout << "Metadata Extracted at: " << output_folder << "metadata.txt";
	    }
	} else {
		std::cerr << "Error: Attribute 'metadata' not found." << std::endl;
	}

    bpReader.Close();
    
    std::cout << "\nImages Recreated at: " << output_folder << std::endl;

}

void deleteExperiment() {
	sqlite3* db;
	int exit = 0;
	exit = sqlite3_open("data.db", &db);

	if (exit) {
		std::cerr << "Error: Can't open database: " << sqlite3_errmsg(db) << std::endl;
		return;
	}

	queryAllData();

	std::string experimentName;
	std::cout << "Enter Experiment Name to Delete: ";
	std::cin >> experimentName;


	if (!checkdb(experimentName)) {
		std::cout << "Experiment Does Not Exist!" << std::endl;		
		return;
	}

	std::string deleteQuery = "DELETE FROM experiment_data WHERE experiment_name = ?;";
	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(db, deleteQuery.c_str(), -1, &stmt, nullptr);

	if (rc != SQLITE_OK) {
		std::cerr << "Error: Failed to prepare delete query: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
		return;
	}

	rc = sqlite3_bind_text(stmt, 1, experimentName.c_str(), -1, SQLITE_STATIC);

	if (rc != SQLITE_OK) {
		std::cerr << "Error: Failed to bind parameters: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_finalize(stmt);
		sqlite3_close(db);
		return;
	}

	rc = sqlite3_step(stmt);

	if (rc != SQLITE_DONE) {
		std::cerr << "Error: Failed to execute delete query: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_finalize(stmt);
		sqlite3_close(db);
		return;
	}

	sqlite3_finalize(stmt);
	sqlite3_close(db);

	// Remove the directory containing the images.bp file
	std::string outputPath = "/home/pbhatia4/Desktop/Adios2C-Implementation/ImageBPFiles/" + experimentName;
	fs::remove_all(outputPath);

	std::cout << "Experiment '" << experimentName << "' Deleted Successfully!" << std::endl;		
		
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: Pass in a flag 1, 2, 3, 4 to make a choice.\n1.) Insert Data\n2.) Query Data\n3.) Extract Data\n4.) Delete Data\n";
        return 1;
    }

    int choice = std::stoi(argv[1]);

    std::cout << "\nSelected Choice: " << choice << "\n";
    std::cout << "-----------------------------" << std::endl;

    if (choice == 1) {
        insertDataAndGetPath();
    } else if (choice == 2) {
        queryAllData();
    } else if (choice == 3) {
        extractImages();
    } else if (choice == 4) {
        deleteExperiment();
    } else {
        std::cerr << "Invalid choice. Please provide a valid flag (1, 2, 3 or 4)\n";
        return 1;
    }

    std::cout << "\nThank you!\nTerminating\n";
    return 0;
    
    // Metadata associated with image variable to be stored in the /bp file 
    // user manually input metaadata for each image, or a description inside a config file / json / yaml file in the database
    // store metadata in both database and adios bp file.
    // design a query interface to query over metadata and experiments.
    // ultimately integrate AI framework
    // 2 paths -- write raw data in adios bp file, another is to insert metadata into databases.
    
    // use protocol buffers to strucutre / serialize metadata.
    // use custom object detection model -- imagenet, huggingface - 
    // Metadata is manually inputted right now, now we should be able to use AI to label the images.
    // ETA before next sem
}

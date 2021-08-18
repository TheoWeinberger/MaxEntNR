/**
 * @file InterpolatorSettings.hpp
 * @author Theo Weinberger
 * @brief This file contains all the relevant functions for reading in the configuration data to run interpolator
 * @version 0.1
 * @date 2021-05-22
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef INTERPOLATORSETTINGS_HPP
#define INTERPOLATORSETTINGS_HPP

/**
 * @brief Method to read file settings into MaxEnt simulation
 * 
 * @param fileName String data containing the name of the file containing configuration data.
 * @param sectionsOut Number of segments of the interpolated data
 * @param errors Whether input data has errors to be interpolated
 * @param rangeTimes Factor by which to scale the max range by
 * @param accountError Whether to accout for experimnetal limitation in the data
 * @param errorKnown Whether the experimental limit of the apparatus is known
 * @param resLimit What the experimnetal limit of the apparatus is 
 * @return int Exit Code
 */
int ReadFile(const std::string& fileName, int& sectionsOut, bool& errors, double& rangeTimes, bool& accountError, bool& errorKnown, double& resLimit);

#endif
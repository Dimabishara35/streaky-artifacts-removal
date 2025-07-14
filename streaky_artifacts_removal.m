function [cleaned_kspace, streaky_coil_indices] = streaky_artifacts_removal(kspace_slice, slice_idx, frame_range)
% REMOVE_STREAKY_ARTIFACTS Removes streaky artifacts from single-slice k-space MRI data
%
% This function identifies and removes streaky artifacts in k-space data for a single slice by:
% 1. Applying a high-pass filter to isolate high-frequency components
% 2. Thresholding and computing energy ratios for each coil to identify streaky coils
% 3. Cleaning data from coils that exceed the streak threshold
%
% Inputs:
%   kspace_slice - 4D k-space data for ONE SLICE [x, y, time, coils]
%   slice_idx    - Index of the current slice being processed (for tracking)
%   frame_range  - Range of time frames to process from the slice (# of frames that meets Nyquist condition)
%
% Outputs:
%   cleaned_kspace       - 4D k-space data for the slice with streaky artifacts removed
%   streaky_coil_indices - Indices of coils identified as having streaky artifacts
%
% Example:
%   [clean_slice, streaky_coils] = remove_streaky_artifacts(kspace_single_slice, 5, 15:20);

    %% Input validation and data extraction
    if nargin < 3
        error('Not enough input arguments. Required: kspace_slice, slice_idx, frame_range');
    end
    
    % Extract data for specified frames from the single slice
    cleaned_kspace = kspace_slice(:, :, frame_range, :);
    [num_x, num_y, num_frames, num_coils] = size(cleaned_kspace);
    
    %% Filter design parameters
    filter_diameter = num_x / 4;
    filter_radius = filter_diameter / 2;
    transition_width = filter_radius * 0.2; % 20% of radius for smooth transition
    
    %% Create high-pass filter
    % Generate coordinate meshgrid centered at image center
    [X, Y] = meshgrid(1:num_x, 1:num_y);
    center_x = num_x / 2;
    center_y = num_y / 2;
    
    % Calculate distance from center for each pixel
    distance_from_center = sqrt((X - center_x).^2 + (Y - center_y).^2);
    
    % Create sigmoid-based high-pass filter
    highpass_filter = 1 ./ (1 + exp(-(distance_from_center - filter_radius) / transition_width))';
    
    % Zero out specific boundary region
    boundary_range = center_x-10 : center_x+10;
    highpass_filter(boundary_range, :) = 0;
    
    % Extend filter across all time frames
    filter_all_frames = repmat(highpass_filter, 1, num_frames);
    
    %% Initialize persistent variables for slice processing
    persistent energy_per_coil streaky_coils percentile_threshold
    
    if isempty(energy_per_coil)
        energy_per_coil = [];
        streaky_coils = [];
        percentile_threshold = [];
    end
    
    %% Calculate threshold from first frame
    fprintf('Processing slice %d: Calculating artifact threshold...\n', slice_idx);
    
    first_frame_data = squeeze(cleaned_kspace(:, :, 1, :));
    percentile_values = zeros(1, num_coils);
    
    % Calculate 95th percentile of filtered high-frequency content for each coil
    for coil_idx = 1:num_coils
        filtered_data = first_frame_data(:, :, coil_idx) .* highpass_filter;
        percentile_values(coil_idx) = prctile(abs(filtered_data(:)), 98);
    end
    
    % Use median as robust threshold estimate
    percentile_threshold(slice_idx) = median(percentile_values);
    
    %% Process all frames and coils
    fprintf('Identifying streaky artifacts across %d frames and %d coils...\n', num_frames, num_coils);
    
    % Reshape data for efficient processing
    slice_data_reshaped = reshape(cleaned_kspace, [num_x, num_y * num_frames, num_coils]);
    cleaned_slice_reshaped = zeros(size(slice_data_reshaped));
    
    % Pre-allocate arrays
    energy_per_coil(slice_idx, 1:num_coils) = 0;
    matrix_size = num_x * num_y;
    
    tic;
    
    % Process each coil
    for coil_idx = 1:num_coils
        % Apply high-pass filter to identify high-frequency content
        filtered_coil_data = slice_data_reshaped(:, :, coil_idx) .* filter_all_frames;
        
        % Create binary mask for pixels exceeding threshold
        artifact_mask = abs(filtered_coil_data) > percentile_threshold(slice_idx);
        
        % Calculate energy (number of pixels exceeding threshold in first frame)
        energy_per_coil(slice_idx, coil_idx) = sum(sum(artifact_mask(:, 1:num_y)));
        
        % Clean data by zeroing pixels identified as artifacts
        cleaned_slice_reshaped(:, :, coil_idx) = (1 - artifact_mask) .* slice_data_reshaped(:, :, coil_idx);
    end
    
    %% Identify streaky coils based on energy threshold
    energy_ratio = (squeeze(energy_per_coil(slice_idx, :))') ./ matrix_size;
    streak_threshold = 0.1; % 10% threshold for identifying streaky coils
    
    streaky_coil_indices = find(energy_ratio >= streak_threshold);
    
    if ~isempty(streaky_coil_indices)
        fprintf('Found %d streaky coils: %s\n', length(streaky_coil_indices), mat2str(streaky_coil_indices));
        
        % Store streaky coil information
        streaky_coils(slice_idx, 1:length(streaky_coil_indices)) = streaky_coil_indices;
        
        % Apply cleaning only to identified streaky coils
        for coil_idx = streaky_coil_indices
            slice_data_reshaped(:, :, coil_idx) = cleaned_slice_reshaped(:, :, coil_idx);
        end
    else
        fprintf('No streaky artifacts detected.\n');
    end
    
    %% Reshape back to original dimensions
    cleaned_kspace = reshape(slice_data_reshaped, [num_x, num_y, num_frames, num_coils]);
    
    elapsed_time = toc;
    fprintf('Artifact removal completed in %.4f seconds\n', elapsed_time);
    
end
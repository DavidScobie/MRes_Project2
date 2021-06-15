tot_folder = uigetdir;
tot_folder = [tot_folder];

all_filenames = dir([tot_folder,'/*.ima']);
no_files = length(all_filenames);

unique_slice_no = strings;
for i = 1:no_files
%     filename = 'SET3_1.MR.0040.*'
    filename = all_filenames(i);
    dot_indics = strfind(filename.name,'.');
    dot_2_pos = dot_indics(2);
    dot_3_pos = dot_indics(3);
    slice_string = filename.name(dot_2_pos+1:dot_3_pos-1);
    
    if any(unique_slice_no(:) == slice_string)
        % do nothing
    else
        unique_slice_no(length(unique_slice_no)+1) = slice_string;
    end

end

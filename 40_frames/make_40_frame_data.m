function [fram_40_data] = make_40_frame_data(rr_int, scan_time, new_dat_final)

first_frame_no = randi(rr_int);
full_seq_length = first_frame_no + scan_time;
n_c_cycles = ceil(full_seq_length ./ rr_int);
long_data = repmat(new_dat_final,[1 1 n_c_cycles]);
first_slice = ceil((first_frame_no./rr_int).*(size(new_dat_final,3)));
fram_40_data = long_data(:,:,first_slice:first_slice+39);

end


sed -i 's/hop_count++;/hop_count++; if(query_id==0) printf("Query0 hop %d pop id %d dist %f\\n", hop_count, current.id, current.dist);/g' cude_version/gpu_search_updated.cuh

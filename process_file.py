import re
import json

def process_file(file_path):
    instances = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Read the first row for number of jobs per instance
        num_jobs_per_instance = int(lines[0].strip())

        # Read the 4th row (index 3 because Python is zero-based)
        row = lines[3].strip()

        if row.startswith('{') and row.endswith('}'):
            row = row[1:-1]  # remove outermost braces
        
        # Use a regex to find top-level sets of braces
        pattern = re.compile(r'\{((?:[^{}]|\{[^{}]*\})*)\}')
        instances_raw = pattern.findall(row)

        # Now parse each instance
        parsed_instances = []
        for instance in instances_raw:
            tuples = re.findall(r'\{([^{}]*)\}', instance)
            for t in tuples:
                numbers = [float(x) for x in t.split(',')]
                parsed_instances.append(tuple(numbers))

        # Now parsed_instances is a flat list of tuples (each with 6 values)
        # Divide them into instances with num_jobs_per_instance each
        num_total_jobs = len(parsed_instances)
        num_instances = num_total_jobs // num_jobs_per_instance

        for i in range(num_instances):
            jobs = []
            start_index = i * num_jobs_per_instance
            end_index = start_index + num_jobs_per_instance

            for j in range(start_index, end_index):
                job_tuple = parsed_instances[j]
                job = {
                    "a": job_tuple[0],
                    "b": job_tuple[1],
                    "c": job_tuple[2],
                    "rda": int(job_tuple[3]),
                    "rdb": int(job_tuple[4]),
                    "dd": int(job_tuple[5])
                }
                jobs.append(job)

            instances.append({
                "num_jobs": num_jobs_per_instance,
                "jobs": jobs
            })

    # Save to a JSON file
    with open("instances2.json", "w") as out_file:
        json.dump(instances, out_file, indent=4)

    print(f"Processed {len(instances)} instances and saved to 'instances2.json'.")
    
    return instances
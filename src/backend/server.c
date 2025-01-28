#include <stdio.h>
#include <string.h>
#include <strings.h>       
#include <sys/socket.h>    
#include <stdlib.h>
#include <netinet/in.h>    
#include <regex.h>         
#include <fcntl.h>         
#include <sys/stat.h>      
#include <unistd.h>        
#include <pthread.h>      


#define PORT 8080
#define BUFFER_SIZE 104857600  //1MB buffer


int server_fd;                    
int shak = 0;                          //flag to indicate if shak.html page is shown 
int gpt = 0;                           //flag to indicate if gpt.html is shown
//the point of hardcoding these is so that the right file can be served based on which file is being shown
//whenever the data is entered in the form input field, since all this is stateless, we have to add some state
//on our own :(


struct sockaddr_in server_addr;   


const char *get_file_extension(const char *filename_url_decoded) {
    const char *dot = strrchr(filename_url_decoded, '.');
    if (!dot || dot == filename_url_decoded) {  
        return "";
    }
    return dot + 1;
}


const char *get_mime_type(const char *file_extension) {
    if (strcasecmp(file_extension, "html") == 0 || strcasecmp(file_extension, "htm") == 0) {
        return "text/html";
    }else if (strcasecmp(file_extension, "css") == 0) {
        return "text/css";
    }   //to also handle css files
     else if (strcasecmp(file_extension, "txt") == 0) {
        return "text/plain";
    } else if (strcasecmp(file_extension, "jpg") == 0 || strcasecmp(file_extension, "jpeg") == 0) {
        return "image/jpeg";
    } else if (strcasecmp(file_extension, "png") == 0) {
        return "image/png";
    } else {
        return "application/octet-stream";
    }
}


char *url_decode(const char *url_filename) {
    size_t src_len = strlen(url_filename);
    char *decoded = (char *)malloc(src_len + 1);  //extra 1 for additional null terminator
    size_t decoded_len = 0;

    //decode %2x to hex, since in url encoded req, space becomes %20 and ! becomes %21
    for (size_t i = 0; i < src_len; i++) {
        if (url_filename[i] == '%' && i + 2 < src_len) {    
            int hex_val;
            sscanf(url_filename + i + 1, "%2x", &hex_val);    
            decoded[decoded_len++] = hex_val;
            i += 2;
        }else {
            decoded[decoded_len++] = url_filename[i];
        }
    }
    decoded[decoded_len] = '\0';
    return decoded;
}


void build_http_response(char *filename, const char *file_ext, char *response, size_t *response_len) {
    //build http header
    printf("REQUESTED FILE: %s\n", filename);
    const char *mime_type = get_mime_type(file_ext);   //gets the mime of the file based on its extention
    char *header = (char *)malloc(sizeof(char) * BUFFER_SIZE);
    snprintf(header, BUFFER_SIZE, 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: %s\r\n"
            "\r\n", mime_type);
    
    //try to open the file requested by the client
    int file_fd = open(filename, O_RDONLY);   
    //if file not found
    if (file_fd == -1) {
        snprintf(response, BUFFER_SIZE, 
                "HTTP/1.1 404 Not Found\r\n"
                "Content-Type: text/plain\r\n"
                "\r\n"
                "404 Not Found"
                );
        *response_len = strlen(response);
        return;
    }

    //get file size for content length
    struct stat file_stat;    
    fstat(file_fd, &file_stat);   
    off_t file_size = file_stat.st_size; 

    //copy header to response buffer
    *response_len = 0;
    memcpy(response, header, strlen(header));
    *response_len += strlen(header);

    ssize_t bytes_read;
    while ((bytes_read = read(file_fd, response + *response_len, BUFFER_SIZE - *response_len)) > 0) {
        *response_len += bytes_read;
    }
    free(header);
    close(file_fd);
}


void *handle_client(void *arg) {
    int client_fd = *((int *)arg);
    char *buffer = (char *)malloc(BUFFER_SIZE * sizeof(char));   
    // char *buffer_ = (char *)malloc(BUFFER_SIZE * sizeof(char));   //part of legacy code

    //recieve the req data from the client and store it in the buffer
    ssize_t bytes_received = recv(client_fd, buffer, BUFFER_SIZE, 0);
    if (bytes_received > 0) {
        memcpy(buffer, buffer, bytes_received);
        buffer[bytes_received] = '\0';
        //see if req is GET
        regex_t regex;    
        regcomp(&regex, "^GET /([^ ]*) HTTP/1", REG_EXTENDED);
        regmatch_t matches[2];
        //regmatch_t matches_referer[2];   //for the referer, not needed now tho
        printf("BUFFER: %s\n", buffer);
        if (regexec(&regex, buffer, 2, matches, 0) == 0) {   
            buffer[matches[1].rm_eo] = '\0'; 
            char *url_encoded_filename = buffer + matches[1].rm_so*sizeof(char);    
            if (strcmp(url_encoded_filename, "shak.html")) {shak = 1; gpt = 0;}    //setting up the shak.html flag
            else if (strcmp(url_encoded_filename, "gpt.html")) {gpt = 1; shak = 0;} //for gpt.html
            //both flags cant be active at the same time, design choice
            printf("URL ENCODED FILENAME: %s\n", url_encoded_filename);

            //the case when a form data is present in the GET req, which is indicated
            //by the presence of the word "submit?prompt="
            if (strstr(url_encoded_filename, "submit?prompt=") != NULL) {
                //code to extract the prompt and save it to a text file
                char *prompt = (char *)(url_encoded_filename + 14);   //since the text "submit?prompt="
                //is fixed we just move the pointer to the start of the actual prompt,
                //where 14 is the index from where the prompt starts in the url_encoded_filenname
 
                //process the prompt
                int counter = 0;
                while (prompt[counter] != '\0') {
                    if (prompt[counter] == '+') {
                        prompt[counter] = ' ';
                    }counter++;
                }
                printf("PROMPT: %s\n", prompt);

                //setting the flags
                if (shak == 1) {
                    url_encoded_filename = "shak.html";
                    pid_t pid = fork();  
                    if (pid == 0) {
                        //system call to run the python script
                        char command[256];
                        int ran = snprintf(command, sizeof(command), "python3 shakgeneration.py 40 %s", prompt);
                        system(command);
                        exit(0);
                        if (ran < 0) {
                            fprintf(stderr, "error in snprintf(), string could not be formatted\n");
                            exit(1);
                        }else {
                            printf("string formatted, command executing!\n");
                        }
                        int exe = system(command);
                        if (exe == -1) {
                            fprintf(stderr, "error executing command\n");
                            exit(1);
                        }printf("child process finsihed executing python script!\n");
                    }else if (pid > 0) {
                        printf("parent process running\n");
                        FILE *file_res = fopen("response.txt", "r");
                        
                        if (!file_res) {
                            perror("failed to open file\n");
                        }
                        char c;
                        while (1) {
                            while ((c = fgetc(file_res)) != EOF) {
                                putchar(c);
                                fflush(stdout);
                            }
                            usleep(50000);           //delay of 50ms, this is needed as without it
                            //the parent process will loop continuously and read from the file wihtout any pause
                            //hence this delay ensures that thre parent checks the file at reasonable intervals
                            //allowing the child to also write the data to the file first before the parent 
                            //attempts to read from it
                            clearerr(file_res);      //when the parent process reaches the EOF, it needs to 
                            //reset EOF as data is being written simultaneosuly by the child process, hence it 
                            //clears the EOF, so that its further able to read from the file later on

                            //check if the child process has exited
                            int status;
                            if (waitpid(pid, &status, WNOHANG) > 0) {
                                if (WIFEXITED(status)) {
                                    printf("child process exited with status %d\n", WEXITSTATUS(status));
                                    break;
                                }
                            }//WNOHANG flag indicates that waitpd shld return immediately if the child
                            //hasnt exited yet, and WIFEXITED() method returns true if the child terminated 
                            //normally(via exit() or return)
                        }fclose(file_res);
                    }else {
                        fprintf(stderr, "fork failed\n");
                        exit(1);
                    }
                }else if (gpt == 1) {
                    url_encoded_filename = "gpt.html";
                }
            }
            char *file_name = url_decode(url_encoded_filename);     
            printf("DECODED FILENAME: %s\n", file_name);

            //get file extension
            char file_ext[32];  
            strcpy(file_ext, get_file_extension(file_name));       

            //build http response
            char *response = (char *)malloc(BUFFER_SIZE * 2 * sizeof(char));
            size_t response_len;
            build_http_response(file_name, file_ext, response, &response_len);

            //send http res to client 
            send(client_fd, response, response_len, 0);

            //incase of a prompt coming in,

            free(response);
            free(file_name);
        }
        regfree(&regex);
    }
    close(client_fd);
    free(arg);
    free(buffer);
    return NULL;
}


int main() {
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE); 
    }

    //configuring the socket
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;  
    server_addr.sin_port = htons(PORT);       

    int opt = 1;  
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt failed");
        exit(EXIT_FAILURE); 
    }
    
    //bind socket to port
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr))) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    //listen for incoming connections
    if (listen(server_fd, 7) < 0) {   //7 is the max number of pending connections allowed
        perror("listen failed");
        exit(EXIT_FAILURE);
    }
   while (1) {
    //client info
    struct sockaddr_in client_addr;  //will store the clients IP address and port number
    // socklen_t client_addr_len = sizeof(client_addr);
    int *client_fd = malloc(sizeof(int));   //this stores the file descriptor for the client socket
    socklen_t client_addr_len = sizeof(client_addr);
    //accept the client conn
    if ((*client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_addr_len)) < 0) {
        perror("accept failed");
        exit(EXIT_FAILURE);
    }     
       pthread_t thread_id;
       pthread_create(&thread_id, NULL, handle_client, (void *)client_fd);
       pthread_detach(thread_id); 
   }
    return 0;
}
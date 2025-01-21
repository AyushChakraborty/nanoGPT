//the server script

#include <stdio.h>
#include <string.h>
#include <strings.h>       //needed for strcasecmp()
#include <sys/socket.h>    //for socket related functions like socket(), bind(), listen(), accept()
#include <stdlib.h>
#include <netinet/in.h>    //for internet specific address family and structures like sockaddr_in
#include <regex.h>         //for regex_t and regex functions
#include <fcntl.h>         //for file control options like O_RDONLY, stands for file control
#include <sys/stat.h>      //for struct stat and file status functions
#include <unistd.h>        //needed for read() for files
#include <pthread.h>       //needed for thread related functions

#define PORT 8080
#define BUFFER_SIZE 104857600  //1MB buffer

int server_fd;                     //file descriptor for server socket, this uniquely identifies the server socket

struct sockaddr_in server_addr;    //struct which contains the server address info
//sockaddr_in is for IPv4, sockaddr_in6 is for IPv6

const char *get_file_extension(const char *filename_url_decoded) {
    /*given image.png  it returns pointer to png
    */
    const char *dot = strrchr(filename_url_decoded, '.');   //returns the last pointer to where
    //'.' occured
    if (!dot || dot == filename_url_decoded) {  //the second case is for cases like
    //.hiddenfile, in this, it has no extention but this is actually the filename
    //hence it rightfully returns ""
        return "";
    }
    return dot + 1;   //returns the pointer to the first char of the extention
}


const char *get_mime_type(const char *file_extension) {
    /*given the file extension itself, find the mime type which stands for 
    //multipurpose internet mail extensions type which is a standard way to 
    //indicate the nature and format of a file or data, here it is used 
    //in the header file sent by the server to the browser so that the client
    //knows how to handle the specific type of data being sent to it as a response
    the form of MIME tyoe is type/subtype
    */
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
    /*egs working
    "GET /path/to%20file HTTP/1.1" is the url encoded req sent from the client to the req,
    now this func given the url encoded file name which in this case is
    /path/to%20file aims to get the same file name but in url decoded format
    which is /path/to file
    */
    size_t src_len = strlen(url_filename);
    char *decoded = (char *)malloc(src_len + 1);  //extra 1 for additional null terminator
    size_t decoded_len = 0;

    //decode %2x to hex, since in url encoded req, space becomes %20 and ! becomes %21
    for (size_t i = 0; i < src_len; i++) {
        if (url_filename[i] == '%' && i + 2 < src_len) {    
            int hex_val;
            sscanf(url_filename + i + 1, "%2x", &hex_val);    //%2x means that excatly
            //2 chars are read and x means that the format is hexadecimal, and the
            //ascii val is then stored in hex_val, 32 is space and it gets rendered as a " "
            //sscanf() is used to read formatted data from a string, it always returns
            //the decimal val(ascii)
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
    printf("req file: %s\n", filename);
    const char *mime_type = get_mime_type(file_ext);   //gets the mime of the file based on its extention
    char *header = (char *)malloc(sizeof(char) * BUFFER_SIZE);
    snprintf(header, BUFFER_SIZE, 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: %s\r\n"
            "\r\n", mime_type);
    
    //try to open the file requested by the client
    int file_fd = open(filename, O_RDONLY);   //this flag opens the file in read only mode
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
    struct stat file_stat;    //this struct contains info such as file size st_size
    //permissions st_mode, creation/modification time st_ctime, st_mtime
    fstat(file_fd, &file_stat);   //this populates struct stat variable file_stat
    //with the open file descriptor
    off_t file_size = file_stat.st_size;   //off_t type is a data type for file sizes
    //and offsets 

    //copy header to response buffer
    *response_len = 0;
    memcpy(response, header, strlen(header));
    *response_len += strlen(header);

    //copy file to response buffer, where the response is actually a pointer to thay buffer hence
    //the same response buffer that is used in handle_client func is also used here
    ssize_t bytes_read;
    while ((bytes_read = read(file_fd, response + *response_len, BUFFER_SIZE - *response_len)) > 0) {
        //where response + *response_len is to move the response buffer's pointer to the location after
        //adding the header to it, and the space left is BUFFER_SIZE - *repsonse_len
        *response_len += bytes_read;
    }
    free(header);
    close(file_fd);
}


void *handle_client(void *arg) {
    //point of this func is to handle the comm with the client in the thread that gets creates
    //by the server socket when the client wants to send a req to the server
    int client_fd = *((int *)arg);
    char *buffer = (char *)malloc(BUFFER_SIZE * sizeof(char));   //allocating memory for the buffer
    char *buffer_ = (char *)malloc(BUFFER_SIZE * sizeof(char));


    //recieve the req data from the client and store it in the buffer
    ssize_t bytes_received = recv(client_fd, buffer, BUFFER_SIZE, 0);
    if (bytes_received > 0) {
        memcpy(buffer_, buffer, bytes_received);
        buffer_[bytes_received] = '\0';
        //see if req is GET
        regex_t regex;    //is used to hold the compiled regular exp, which is later
        //used to match against the http req received from the client 
        regex_t referer_regex;   //regex for the referer
        regcomp(&regex, "^GET /([^ ]*) HTTP/1", REG_EXTENDED);   //the compiled format is then stored in regex var
        //the format we are looking for is of type say "GET /index.html HTTP/1.1"
        //also REG_EXTENDED flag modifies how the regex is interpreted, with this flag, POSIX extended regex is used
        //which allows more modern, flexible syntax while specifying the regex pattern to be compiled
        regmatch_t matches[2];
        regmatch_t matches_referer[2];   //for the referer
        printf("req from client is: %s\n", buffer_);
        if (regexec(&regex, buffer_, 2, matches, 0) == 0) {   //if the req contents in the buffer matches that in
        //the pattern compiled into regex variable, then do the following

            //extract the filename from req and decode url
            buffer_[matches[1].rm_eo] = '\0';   //setting the end offset of the captured group to be null byte,
            //essentially isolating this substring in buffer
            const char *url_encoded_filename = buffer_ + matches[1].rm_so*sizeof(char);    //moves the pointer to the start
            //of the matched substring, as buffer itself is a pointer to the actual buffer object
            printf("url encoded filename: %s\n", url_encoded_filename);

            //the case when a form data is present in the GET req, which is indicated
            //by the presence of the word "submit?prompt="
            printf("Checking if 'submit?prompt=' is in: %s\n", url_encoded_filename);
            if (strstr(url_encoded_filename, "submit?prompt=") != NULL) {
                printf("BUFFER: %s\n", buffer);
                regcomp(&referer_regex, "Referer: [^ ]*/([^ ]*\\.html)", REG_EXTENDED);
                int check = regexec(&referer_regex, buffer, 2, matches_referer, 0);
                printf("regex info: %d\n", check);
                if (regexec(&referer_regex, buffer, 2, matches_referer, 0) == 0) {
                    printf("in intended area\n");
                    buffer[matches_referer[1].rm_eo] = '\0';
                    url_encoded_filename = buffer + matches_referer[1].rm_so*sizeof(char);
                    printf("url encoded filename from form data: %s\n", url_encoded_filename);
                }regfree(&referer_regex);
            }
            // buffer = buffer_;
            char *file_name = url_decode(url_encoded_filename);     //this will get the
            //file name now anyways
            printf("decoded file name: %s\n", file_name);


            //get file extension
            char file_ext[32];    //copy the file extension obtained from the get_file_extension(file_name) func
            //to file_ext variable made here
            strcpy(file_ext, get_file_extension(file_name));       


            //build http response
            char *response = (char *)malloc(BUFFER_SIZE * 2 * sizeof(char));
            size_t response_len;
            build_http_response(file_name, file_ext, response, &response_len);


            //send http res to client 
            send(client_fd, response, response_len, 0);

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
    //create server socket, AF_INET specifies the Address Family IPv4 in this case, 
    //SOCK_STREAM specifies the type of socket, in this case TCP which is a protocol
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE); 
    }

    //configuring the socket
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;  //INADDR_ANY allows the server to accept
    //connections on any of the machines' available network interfaces like ethernet, WIFI
    server_addr.sin_port = htons(PORT);       //htons() converts the port number to network byte order
    //which is just big endian representation of the port number

    int opt = 1;   //the value that will be be set for SO_REUSEADDR
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt failed");
        exit(EXIT_FAILURE); 
    }//this allows the server to bind to the same socket even if its is busy, this is useful
    //when the server is opened and closed very fast and hence the socket
    //is not released from the server
    
    //bind socket to port
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr))) {
        //the bind func binds the server socket to the port number and the IP address
        //here typecasting is done as bind expects a generic pointer to a sockaddr struct
        //not some specific IPv4 or IPv6 struct
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    //listen for incoming connections
    if (listen(server_fd, 7) < 0) {   //7 is the max number of pending connections that can be queued
    //before the server starts rejecting new connections, this value is a limit on how 
    //many clients can wait to be served while the server processes other clients
        perror("listen failed");
        exit(EXIT_FAILURE);
    }

    //now we handle incoming client connections 
    /*when a client connects to the server(through the socket), the server accepts the connection
    and creates a new thread to handle the clients http req, this way the server can handle multiple
    clients concurrently
    */
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
       //this was handling the client connection, now that its established, a thread for this 
       //client is created by the server to handle the http req and resps
       pthread_t thread_id;
       pthread_create(&thread_id, NULL, handle_client, (void *)client_fd);
       //NULL here means that the thread will be created with default attributes like 
       //default stack sizem scheduling policy etc, and handle_client is the func 
       //which will be executed when the thread is started, its also known as the
       //thread routine/entry point
       //again also typecasted in (void *) as the thread routine expects a void pointer
       //wihtin this API call, client_fd is passed as an argument to the thread routine
       //which is handle_client hence the typecast is needed as handle_client's signature
       //is void *handle_client(void *arg)
       pthread_detach(thread_id); 
   }


    return 0;
}
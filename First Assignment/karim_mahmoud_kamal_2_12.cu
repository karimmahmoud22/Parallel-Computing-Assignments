// Name: Karim Mahmoud Kamal Mohmed
// Section: 2
// B.N: 12

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()
{
    int number_of_rows, number_of_columns;
    scanf("%d %d", &number_of_rows, &number_of_columns);
    
    int sum = 0;
    int **matrix = (int **)malloc(sizeof(int *) * number_of_rows);

    if (matrix == NULL)
    {
        printf("%s", "Failed to allocate memory for the matrix");
        return -1;
    }

    for (int i = 0; i < number_of_rows; i++)
    {
        matrix[i] = (int *)malloc(sizeof(int) * number_of_columns);
        if (matrix[i] == NULL)
        {
            printf("Failed to allocate memory for the row number: %d\n", i);
            return -1;
        }
    }

    for (int i = 0; i < number_of_rows; ++i){
        for (int j = 0; j < number_of_columns; ++j){
            int num;
            scanf("%d", &num);
            matrix[i][j] = num;
        }
    }

    for (int i = 0; i < number_of_columns; i++)
    {
        char *result = (char *)malloc(sizeof(char) * 300);
        strcpy(result, "");
        char temp[200];
        for (int j = 0; j < number_of_rows; j++)
        {
            sprintf(temp, "%d", matrix[j][i]);
            strcat(result, temp);
        }
        sum += atoi(result);
        free(result);
    }

    printf("%d", sum);

    for (int i = 0; i < number_of_rows; ++i)
        free(matrix[i]);
    free(matrix);

    return 0;
}

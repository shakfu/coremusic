#include <CoreFoundation/CoreFoundation.h>
#include <string.h>
#include <stdio.h>

int main()
{

    char externalKey[512] = {0};

    strcpy(externalKey, "hello there bro");
    printf("key: %s\n", externalKey);

    /* First of all, we need a C array to store our dictionary keys */
    CFStringRef keys[ 2 ];

    /*
     * Let's create the dictionary keys. First key is straightforward because
     * of the CFSTR macro, while the second one is less...
     */
    keys[ 0 ] = CFSTR( "key-1" );
    keys[ 1 ] = CFStringCreateWithCString
    (
        ( CFAllocatorRef )NULL,
        externalKey,
        kCFStringEncodingUTF8
    );

    // /* Let's create the data object from the external byte buffer */
    // CFDataRef data = CFDataCreate
    // (
    //     ( CFAllocatorRef )NULL,
    //     ( const UInt8 * )externalByteBuffer,
    //     ( CFIndex )externalByteBufferLength
    // );

    // /*
    //  * Now, let's create some number object. Note that we need a temporary
    //  * variable, as we need to pass an address of a primitive type...
    //  */
    // int tempInt        = 42;
    // CFNumberRef number = CFNumberCreate
    // (
    //     ( CFAllocatorRef )NULL,
    //     kCFNumberSInt32Type,
    //     &tempInt
    // );

    // /*
    //  * Now, create an URL object. Note that we need to create a temporary
    //  * CFString object
    //  */
    // CFStringRef tempStringURL = CFStringCreateWithCString
    // (
    //     ( CFAllocatorRef )NULL,
    //     externalURL,
    //     kCFStringEncodingUTF8
    // );
    // CFURLRef url = CFURLCreateWithString
    // (
    //     ( CFAllocatorRef )NULL,
    //     tempStringURL,
    //     NULL
    // );

    // /* Before creating the array, we need a C array to store the values */
    // CFTypeRef arrayValues[ 2 ];

    // arrayValues[ 0 ] = number;
    // arrayValues[ 1 ] = url;

    // /* Now we can create the array... */
    // CFArrayRef array = CFArrayCreate
    // (
    //     ( CFAllocatorRef )NULL,
    //     ( const void ** )arrayValues,
    //     2,
    //     &kCFTypeArrayCallBacks
    // );

    // /* Now, of course, we need a C array to store the dictionary values */
    // CFTypeRef values[ 2 ];

    // values[ 0 ] = data;
    // values[ 1 ] = array;

    // /* Finally, we can create our dictionary... */
    // CFDictionaryRef dictionary = CFDictionaryCreate
    // (
    //     ( CFAllocatorRef )NULL,
    //     ( const void ** )keys,
    //     ( const void ** )values,
    //     2,
    //     &kCFTypeDictionaryKeyCallBacks,
    //     &kCFTypeDictionaryValueCallBacks
    // );

    // /* And of course, as we allocated objects, we need to release them... */
    CFRelease( keys[ 1 ] );
    // CFRelease( data );
    // CFRelease( number );
    // CFRelease( tempStringURL );
    // CFRelease( url );
    // CFRelease( array );

    // /* That's it, prints the dictionary and release it */
    // CFShow( dictionary );
    // CFRelease( dictionary );
}

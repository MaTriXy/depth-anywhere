// Standalone build of the Android bindings. Apps can also include :depthanything directly
// (include + projectDir), as the Portable Shaders showcase does.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "depth-anything-android"
include(":depthanything")
